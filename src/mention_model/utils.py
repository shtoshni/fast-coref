
def get_mention_model_name(args):
    model_name_suffix = ""
    model_name_suffix += f'{args.dataset}' + "_"
    model_name_suffix += 'width_' + f'{args.max_span_width}' + "_"
    model_name_suffix += 'mlp_' + f'{args.mlp_size}' + '_'
    # model_name_suffix += 'drop_' + f'{args.dropout_rate}' + '_'
    model_name_suffix += 'model_' + f'{args.model_size}' + '_'
    model_name_suffix += 'emb_' + f'{args.ment_emb}' + '_'
    model_name_suffix += 'type_' + ('spanbert' if args.pretrained_bert_dir else 'bert') + '_'
    model_name_suffix += 'enc_' + f'{args.doc_enc}' + '_'
    model_name_suffix += 'segment_' + f'{args.max_segment_len}' + '_'

    if args.dataset == 'litbank':
        model_name_suffix += 'split_' + f'{args.cross_val_split}' + '_'

    if model_name_suffix[-1] == '_':
        model_name_suffix = model_name_suffix[:-1]

    if args.train_with_singletons and args.dataset == 'ontonotes':
        model_name_suffix += '_singleton'
    elif (not args.train_with_singletons) and args.dataset == 'litbank':
        model_name_suffix += '_no_singleton'

    model_name = "ment_" + model_name_suffix
    return model_name


