import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil


def feature_matching(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    akaze = cv2.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    num_matches = len(matches)

    result_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return num_matches, result_img


def main():
    images = [
        ("img/1-1.jpg", "img/1-2.jpg"),
        ("img/2-1.jpg", "img/2-2.jpg"),
        ("img/3-1.jpg", "img/3-2.jpg"),
    ]

    for i, img in enumerate(images):
        original_img_1 = Image.open(img[0]).resize([1920, 1080])
        original_img_2 = Image.open(img[1]).resize([1920, 1080])

        original_num_matches, original_result_img = feature_matching(
            original_img_1, original_img_2
        )
        grey_world_num_matches, grey_world_result_img = feature_matching(
            to_pil(cca.grey_world(from_pil(original_img_1))), to_pil(cca.grey_world(from_pil(original_img_2)))
        )
        retinex_num_matches, retinex_result_img = feature_matching(
            to_pil(cca.retinex(from_pil(original_img_1))), to_pil(cca.retinex(from_pil(original_img_2)))
        )
        max_white_num_matches, max_white_result_img = feature_matching(
            to_pil(cca.max_white(from_pil(original_img_1))), to_pil(cca.max_white(from_pil(original_img_2)))
        )
        stretch_num_matches, stretch_result_img = feature_matching(
            to_pil(cca.stretch(from_pil(original_img_1))), to_pil(cca.stretch(from_pil(original_img_2)))
        )
        retinex_with_adjust_num_matches, retinex_with_adjust_result_img = feature_matching(
            to_pil(cca.retinex_with_adjust(from_pil(original_img_1))), to_pil(cca.retinex_with_adjust(from_pil(original_img_2)))
        )
        standard_deviation_num_matches, standard_deviation_result_img = feature_matching(
            to_pil(cca.standard_deviation_weighted_grey_world(from_pil(original_img_1))), to_pil(cca.standard_deviation_weighted_grey_world(from_pil(original_img_2)))
        )
        standard_deviation_luminance_num_matches, standard_deviation_luminance_result_img = feature_matching(
            to_pil(cca.standard_deviation_and_luminance_weighted_gray_world(from_pil(original_img_1))), to_pil(cca.standard_deviation_and_luminance_weighted_gray_world(from_pil(original_img_2)))
        )
        luminance_num_matches, luminance_result_img = feature_matching(
            to_pil(cca.luminance_weighted_gray_world(from_pil(original_img_1))), to_pil(cca.luminance_weighted_gray_world(from_pil(original_img_2)))
        )
        automatic_color_equalization_num_matches, automatic_color_equalization_result_img = feature_matching(
            to_pil(cca.automatic_color_equalization(from_pil(original_img_1))), to_pil(cca.automatic_color_equalization(from_pil(original_img_2)))
        )

        plt.figure(figsize=(10, 10))
        plt.subplot(5, 2, 1)
        plt.imshow(cv2.cvtColor(original_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original: {original_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 2)
        plt.imshow(cv2.cvtColor(grey_world_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Grey World: {grey_world_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 3)
        plt.imshow(cv2.cvtColor(retinex_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Retinex: {retinex_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 4)
        plt.imshow(cv2.cvtColor(max_white_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Max White: {max_white_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 5)
        plt.imshow(cv2.cvtColor(stretch_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Stretch: {stretch_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 6)
        plt.imshow(cv2.cvtColor(retinex_with_adjust_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Retinex with Adjust: {retinex_with_adjust_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 7)
        plt.imshow(cv2.cvtColor(standard_deviation_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Standard Deviation: {standard_deviation_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 8)
        plt.imshow(cv2.cvtColor(standard_deviation_luminance_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Standard Deviation Luminance: {standard_deviation_luminance_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 9)
        plt.imshow(cv2.cvtColor(luminance_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Luminance: {luminance_num_matches} matches")
        plt.axis("off")

        plt.subplot(5, 2, 10)
        plt.imshow(cv2.cvtColor(automatic_color_equalization_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Automatic Color Equalization: {automatic_color_equalization_num_matches} matches")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"result/{i + 1}.png")
        plt.close()

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
