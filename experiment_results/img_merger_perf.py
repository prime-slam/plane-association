from PIL import Image

if __name__ == "__main__":
    imgs_path = [
        "assoc_perf_icl_living.png",
        "assoc_perf_icl_office.png",
        "assoc_perf_tum_cabinet.png",
        "assoc_perf_tum_desk.png",
        "assoc_perf_tum_long_office.png",
        "assoc_perf_tum_pioneer.png",
    ]

    imgs = []
    for img_path in imgs_path:
        imgs.append(Image.open(img_path))
    img_size = imgs[0].size

    new_im = Image.new("RGB", (3 * img_size[0], 2 * img_size[1]), (255, 255, 255))

    for i, img in enumerate(imgs):
        x = i % 3 * (img_size[0] + 0)
        y = i // 3 * (img_size[1] + 0)
        new_im.paste(img, (x, y))

    new_im.save("performance.png", "PNG")
