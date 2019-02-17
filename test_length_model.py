import torch
import onmt.utils.length_model
import configargparse
import onmt.opts as opts


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='length_model.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.test_length_model_opts(parser)

    opt = parser.parse_args()

    checkpoint = torch.load(opt.mt_model,
                            map_location=lambda storage, loc: storage)
    vocab = checkpoint['vocab'][1][1]
    onmt.utils.length_model.test(opt, vocab)
