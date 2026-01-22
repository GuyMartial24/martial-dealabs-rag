from pathlib import Path

from indexing.deals_vector_indexing import Config, MigrationPipeline

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    config = Config(
        data_dir=str(PROJECT_ROOT / "data" / "newdeals"),
        batch_size=100,
    )

    pipeline = MigrationPipeline(config)
    pipeline.run(reset=True, create_index=True)


if __name__ == "__main__":
    main()
