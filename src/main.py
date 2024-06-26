import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0] * (mu_g / np.average(nimg[0])), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / np.average(nimg[2])), 255)

    return nimg.transpose(1, 2, 0).astype(np.uint8)


def feature_matching(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    num_matches = len(matches)
    result_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:0],
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
        img1 = cv2.resize(cv2.imread(img[0]), (1280, 720))
        img2 = cv2.resize(cv2.imread(img[1]), (1280, 720))

        original_num_matches, original_result_img = feature_matching(img1, img2)
        gray_scale_num_matches, gray_scale_result_img = feature_matching(
            cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY),
        )
        gray_world_num_matches, gray_world_result_img = feature_matching(
            gray_world(img1), gray_world(img2)
        )

        plt.figure(figsize=(10, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(original_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original: {original_num_matches} matches", fontsize=20)
        plt.axis("off")

        plt.subplot(3, 1, 2)
        plt.imshow(cv2.cvtColor(gray_scale_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Grayscale: {gray_scale_num_matches} matches", fontsize=20)
        plt.axis("off")

        plt.subplot(3, 1, 3)
        plt.imshow(cv2.cvtColor(gray_world_result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Gray World: {gray_world_num_matches} matches", fontsize=20)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"result/{i + 1}.png")
        plt.close()

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
