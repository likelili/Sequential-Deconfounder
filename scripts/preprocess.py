from sequential_deconfounder.data.preprocess import build_argparser, run_pipeline


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
