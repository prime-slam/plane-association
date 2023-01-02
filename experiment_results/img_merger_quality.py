from PIL import Image

if __name__ == "__main__":
    imgs_path = [
        "assoc_IoUWeighted_planes_icl_living.png",
        "assoc_IoUThresholded_planes_icl_living.png",
        "assoc_IoUWeighted_points_icl_living.png",
        "assoc_IoUThresholded_points_icl_living.png",
        "assoc_IoUWeighted_planes_icl_office.png",
        "assoc_IoUThresholded_planes_icl_office.png",
        "assoc_IoUWeighted_points_icl_office.png",
        "assoc_IoUThresholded_points_icl_office.png",
        "assoc_IoUWeighted_planes_tum_cabinet.png",
        "assoc_IoUThresholded_planes_tum_cabinet.png",
        "assoc_IoUWeighted_points_tum_cabinet.png",
        "assoc_IoUThresholded_points_tum_cabinet.png",
        "assoc_IoUWeighted_planes_tum_desk.png",
        "assoc_IoUThresholded_planes_tum_desk.png",
        "assoc_IoUWeighted_points_tum_desk.png",
        "assoc_IoUThresholded_points_tum_desk.png",
        "assoc_IoUWeighted_planes_tum_long_office.png",
        "assoc_IoUThresholded_planes_tum_long_office.png",
        "assoc_IoUWeighted_points_tum_long_office.png",
        "assoc_IoUThresholded_points_tum_long_office.png",
        "assoc_IoUWeighted_planes_tum_pioneer.png",
        "assoc_IoUThresholded_planes_tum_pioneer.png",
        "assoc_IoUWeighted_points_tum_pioneer.png",
        "assoc_IoUThresholded_points_tum_pioneer.png",
    ]
    COLS = 4
    ROWS = 6

    imgs = []
    for img_path in imgs_path:
        imgs.append(Image.open(img_path))
    img_size = imgs[0].size

    new_im = Image.new("RGB", (COLS * img_size[0], ROWS * img_size[1]), (255, 255, 255))

    for i, img in enumerate(imgs):
        x = i % COLS * (img_size[0] + 0)
        y = i // COLS * (img_size[1] + 0)
        new_im.paste(img, (x, y))

    new_im.save("assoc_quality.png", "PNG")
