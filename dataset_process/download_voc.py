from pathlib import Path
from torchvision.datasets import VOCDetection


DATA_ROOT = Path("./data")


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("Downloading VOC2007 trainval...")
    voc2007_trainval = VOCDetection(
        root=DATA_ROOT,
        year="2007",
        image_set="trainval",
        download=True
    )

    print("Downloading VOC2012 trainval...")
    voc2012_trainval = VOCDetection(
        root=DATA_ROOT,
        year="2012",
        image_set="trainval",
        download=True
    )

    print("Downloading VOC2007 test...")
    voc2007_test = VOCDetection(
        root=DATA_ROOT,
        year="2007",
        image_set="test",
        download=True
    )

    print("\nDownload completed.")
    print(f"VOC2007 trainval: {len(voc2007_trainval)} images")
    print(f"VOC2012 trainval: {len(voc2012_trainval)} images")
    print(f"VOC2007 test:     {len(voc2007_test)} images")
    print(
        f"Total training images: "
        f"{len(voc2007_trainval) + len(voc2012_trainval)}"
    )


if __name__ == "__main__":
    main()