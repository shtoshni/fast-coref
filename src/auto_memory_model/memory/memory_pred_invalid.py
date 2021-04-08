import torch
from auto_memory_model.memory import BaseMemory
from pytorch_utils.modules import MLP


class MemoryPredInvalid(BaseMemory):
    def __init__(self, mem_type='unbounded', **kwargs):
        super(MemoryPredInvalid, self).__init__(**kwargs)
        self.mem_type = mem_type
        self.is_mem_bounded = self.max_ents is not None

        if self.is_mem_bounded:
            self.fert_mlp = MLP(input_size=self.mem_size + self.num_feats * self.emb_size,
                                hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                                bias=True, drop_module=self.drop_module)

    def predict_new_or_ignore_learned(self, query_vector, mem_vectors, feature_embs, ment_feature_embs):
        # Fertility Score
        mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        ment_fert_input = torch.unsqueeze(torch.cat([query_vector, ment_feature_embs], dim=-1), dim=0)
        fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)

        fert_scores = self.fert_mlp(fert_input)
        fert_scores = torch.squeeze(fert_scores, dim=-1)

        new_or_ignore_scores = fert_scores

        output = new_or_ignore_scores,
        max_idx = torch.argmax(new_or_ignore_scores).item()
        if max_idx < self.max_ents:
            return output + (max_idx, 'o')
        else:
            # No space - The new entity is not "fertile" enough
            return output + (-1, 'n')

    def predict_new_or_ignore_lru(self, query_vector, mem_vectors,
                                  feature_embs, ment_feature_embs, lru_list):
        lru_cell = lru_list[0]
        mem_fert_input = torch.cat([mem_vectors[lru_cell, :], feature_embs[lru_cell, :]], dim=0)
        ment_fert_input = torch.cat([query_vector, ment_feature_embs], dim=-1)
        fert_input = torch.stack([mem_fert_input, ment_fert_input], dim=0)
        fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)
        output = fert_scores,

        over_max_idx = torch.argmax(fert_scores).item()
        if over_max_idx == 0:
            return output + (lru_cell, 'o')
        elif over_max_idx == 1:
            # No space - The new entity is not "fertile" enough
            return output + (-1, 'n')

    def forward(self, ment_boundaries, mention_emb_list, gt_actions, metadata,
                memory_init=None, teacher_forcing=False):
        # Initialize memory
        if memory_init is not None:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory(**memory_init)
        else:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory()

        coref_new_list, new_ignore_list = [], []
        action_list = []  # argmax actions
        first_overwrite = (True if torch.sum(ent_counter) == 0 else False)

        follow_gt = self.training or teacher_forcing

        # For LRU memory
        lru_list = []
        if self.mem_type == 'lru':
            lru_list = list(range(self.max_ents))

        # print(len(gt_actions), len(mention_emb_list))
        for ment_idx, (ment_emb, (gt_cell_idx, gt_action_str)) in enumerate(zip(mention_emb_list, gt_actions)):
            ment_start, ment_end = ment_boundaries[ment_idx]
            query_vector = ment_emb
            num_ents = mem_vectors.shape[0]
            # metadata['last_action'] = self.action_str_to_idx[last_action_str]
            feature_embs = self.get_feature_embs(ment_start, last_mention_start, ent_counter, metadata)

            if first_overwrite:
                pred_cell_idx, pred_action_str = 0, 'o'
            else:
                coref_new_scores = self.get_coref_new_scores(
                    query_vector, mem_vectors, ent_counter, feature_embs)

                pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores, first_overwrite)
                coref_new_list.append(coref_new_scores)

            if self.is_mem_bounded:
                # Check if the model has reached memory capacity
                if (follow_gt and gt_action_str != 'c' and num_ents == self.max_ents) or \
                        ((not follow_gt) and pred_cell_idx == self.max_ents):
                    # Reached memory capacity
                    if self.mem_type == 'learned':
                        new_or_ignore_scores, pred_cell_idx, pred_action_str = self.predict_new_or_ignore_learned(
                            query_vector, mem_vectors, feature_embs, self.get_ment_feature_embs(metadata))
                        new_ignore_list.append(new_or_ignore_scores)
                    elif self.mem_type == 'lru':
                        new_or_ignore_scores, pred_cell_idx, pred_action_str = self.predict_new_or_ignore_lru(
                            query_vector, mem_vectors, feature_embs, self.get_ment_feature_embs(metadata), lru_list)
                        new_ignore_list.append(new_or_ignore_scores)

            action_list.append((pred_cell_idx, pred_action_str))

            if follow_gt:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

            # last_action_str = action_str

            if first_overwrite and action_str == 'o':
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(query_vector, dim=0)
                ent_counter = torch.tensor([1.0], device=self.device)
                last_mention_start[0] = ment_start
            else:
                # Update the memory
                cell_mask = (torch.arange(0, num_ents, device=self.device) == cell_idx).float()
                mask = torch.unsqueeze(cell_mask, dim=1)
                mask = mask.repeat(1, self.mem_size)

                # print(cell_idx, action_str, mem_vectors.shape[0])
                if action_str == 'c':
                    coref_vec = self.coref_update(mem_vectors, query_vector, cell_idx, ent_counter)
                    if self.training:
                        mem_vectors = mem_vectors * (1 - mask) + mask * coref_vec
                    else:
                        mem_vectors[cell_idx] = coref_vec
                    ent_counter[cell_idx] = ent_counter[cell_idx] + 1
                    last_mention_start[cell_idx] = ment_start

                elif action_str == 'o':
                    if cell_idx == num_ents:
                        # Append the new vector
                        mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)

                        ent_counter = torch.cat([ent_counter, torch.tensor([1.0], device=self.device)], dim=0)
                        last_mention_start = torch.cat(
                            [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0)
                    else:
                        # Replace the cell content tracking another entity
                        if self.training:
                            mem_vectors = mem_vectors * (1 - mask) + mask * torch.unsqueeze(query_vector, dim=0)
                        else:
                            mem_vectors[cell_idx] = query_vector
                        last_mention_start[cell_idx] = ment_start
                        ent_counter = ent_counter * (1 - cell_mask) + cell_mask

            if self.mem_type == 'lru' and action_str in ['o', 'c']:
                # Coref or overwrite was chosen; place the cell_idx in use at the back of the list
                lru_list.remove(cell_idx)
                lru_list.append(cell_idx)

        mem_state = {"mem": mem_vectors, "ent_counter": ent_counter, "last_mention_start": last_mention_start}
        return coref_new_list, new_ignore_list, action_list, mem_state
