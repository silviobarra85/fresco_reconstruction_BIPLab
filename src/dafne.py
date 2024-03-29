import cv2
import os
import math
import time
import argparse

t = time.time()


def angolo(x1, y1, x2, y2):
    flag = 0
    # print("-")
    if (x1 < x2):
        flag = 1
    cc1 = coeff_angolare(x1, y1, x2, y2)
    cc2 = coeff_angolare(x1, y1, x1 + 10, y1)
    acc = (cc1 - cc2) / (1 + cc1 * cc2)
    arct = math.atan(acc)  #  restituisce il valore in radianti
    return math.degrees(arct), flag


def coeff_angolare(x1, y1, x2, y2):
    numeratore = (y2 - y1)
    denominatore = (x2 - x1)
    return numeratore / denominatore


def distanza(x1, y1, x2, y2):
    d12 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return d12


def ricomponi(frammento, immagine_intera, output):
    whole_image = immagine_intera
    img1_filename = frammento

    img1 = cv2.imread(img1_filename, 0)  # queryImage
    img2 = cv2.imread(whole_image, 0)  # trainImage

    img1_alfa = cv2.imread(img1_filename, cv2.IMREAD_UNCHANGED)  # trainImage CON ALPHA CHANNEL

    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)

    # minHessian = 400
    sift = cv2.xfeatures2d_SIFT.create()

    sift_kp1, sift_des1 = sift.detectAndCompute(img1, None)
    sift_kp2, sift_des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    knn_matches = matcher.knnMatch(sift_des1, sift_des2, k=2)

    ratio_thresh = 0.7
    good_matches = []

    idx1 = []  # gli indici dei kp nell'immagine piccola
    idx2 = []  # gli indici dei kp nell'immagine grande

    indice_kp = 0
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            idx1.append(indice_kp)
        indice_kp = indice_kp + 1

    for m in good_matches:
        idx2.append(m.trainIdx)

    dist_minima = 2000
    origine_big = destino_big = 0
    indice_origine = indice_destino = 0
    idx_origine_little = idx_destino_little = 0

    #	for punto1 in good_matches:
    #		origine = punto1.trainIdx
    #		for punto2 in good_matches:
    #			destino = punto2.trainIdx
    #			dist_mom = distanza (surf_kp2[origine].pt[0] ,surf_kp2[origine].pt[1], surf_kp2[destino].pt[0], surf_kp2[destino].pt[1])
    #			if(dist_mom != 0 and dist_mom < dist_minima):
    #				dist_minima = dist_mom
    #				origine_big = origine
    #				destino_big = destino
    #				idx_origine_little = indice_origine
    #				idx_destino_little = indice_destino
    #			indice_destino = indice_destino + 1
    #		indice_origine = indice_origine + 1
    #		indice_destino  = 0

    for punto1 in good_matches:
        origine = punto1.trainIdx
        for punto2 in good_matches:
            destino = punto2.trainIdx
            dist_2 = distanza(sift_kp2[origine].pt[0], sift_kp2[origine].pt[1], sift_kp2[destino].pt[0],
                              sift_kp2[destino].pt[1])
            dist_1 = distanza(sift_kp1[idx1[indice_origine]].pt[0], sift_kp1[idx1[indice_origine]].pt[1],
                              sift_kp1[idx1[indice_destino]].pt[0], sift_kp1[idx1[indice_destino]].pt[0])

            dist_1 = int(dist_1)
            dist_2 = int(dist_2)

            # print(dist_1 , "  -- distanze --  " , dist_2)
            if (dist_1 != 0 and dist_2 != 0 and dist_1 == dist_2):
                # print(dist_1 , "  -- distanze --  " , dist_2)
                # print("------------------ ENTER -----------------------")
                # dist_minima = dist_mom
                origine_big = origine
                destino_big = destino
                idx_origine_little = indice_origine
                idx_destino_little = indice_destino
            indice_destino = indice_destino + 1
        indice_origine = indice_origine + 1
        indice_destino = 0

    # ----- SCALE

    big_1x = sift_kp2[origine_big].pt[0]
    big_1y = sift_kp2[origine_big].pt[1]
    big_2x = sift_kp2[destino_big].pt[0]
    big_2y = sift_kp2[destino_big].pt[1]

    d_big = distanza(big_1x, big_1y, big_2x, big_2y)

    # DISTANZA S1 - S2  - little image
    little_1 = sift_kp1[idx1[idx_origine_little]]
    little_2 = sift_kp1[idx1[idx_destino_little]]

    little_1x = little_1.pt[0]
    little_1y = little_1.pt[1]
    little_2x = little_2.pt[0]
    little_2y = little_2.pt[1]

    d_little = distanza(little_1x, little_1y, little_2x, little_2y)

    nuova_immagine = img1

    # ----- ROTATE

    angolo_little, f1 = angolo(little_1x, little_1y, little_2x, little_2y)
    # print("angolo little . " , angolo_little)

    angolo_big, f2 = angolo(big_1x, big_1y, big_2x, big_2y)
    # print("angolo big . " , angolo_big)

    if ((f1 - f2) != 0):
        rotazione = (180 + angolo_little) - angolo_big
    else:
        rotazione = angolo_little - angolo_big

    rows, cols = nuova_immagine.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotazione, 1)
    dst = cv2.warpAffine(nuova_immagine, M, (cols, rows))

    img1_alfa = cv2.warpAffine(img1_alfa, M, (cols, rows))

    # ----- INSERT

    s_img = img1_alfa
    l_img = cv2.imread(whole_image, cv2.IMREAD_UNCHANGED)  #  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img1 = dst

    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    sift2 = cv2.xfeatures2d_SIFT.create()
    # surf = cv2.xfeatures2d.SIFT_create()

    sift2_kp1, sift2_des1 = sift2.detectAndCompute(img1, None)
    sift2_kp2, sift2_des2 = sift2.detectAndCompute(img2, None)
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(sift2_des1, sift2_des2, k=2)

    # -- Filter matches using the Lowe's ratio test
    # ratio_thresh = 0.9
    ratio_thresh = 0.7
    good_matches = []

    idx1 = []  # gli indici dei kp nell'immagine piccola
    idx2 = []  # gli indici dei kp nell'immagine grande

    indice_kp = 0

    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            idx1.append(indice_kp)
            indice = m.trainIdx
        indice_kp = indice_kp + 1

    for m in good_matches:
        idx2.append(m.trainIdx)

    dist_minima = 2000
    origine_big = destino_big = 0
    indice_origine = indice_destino = 0
    idx_origine_little = idx_destino_little = 0

    for punto1 in good_matches:
        origine = punto1.trainIdx
        for punto2 in good_matches:
            destino = punto2.trainIdx
            dist_mom = distanza(sift2_kp2[origine].pt[0], sift2_kp2[origine].pt[1], sift2_kp2[destino].pt[0],
                                sift2_kp2[destino].pt[1])
            dist_little_mom = distanza(sift2_kp1[idx1[indice_origine]].pt[0], sift2_kp1[idx1[indice_origine]].pt[1],
                                       sift2_kp1[idx1[indice_destino]].pt[0], sift2_kp1[idx1[indice_destino]].pt[1])
            # if(int(dist_mom) == int(dist_little_mom)):
            if (dist_mom != 0 and dist_mom < dist_minima):
                dist_minima = dist_mom
                origine_big = origine
                destino_big = destino
                idx_origine_little = indice_origine
                idx_destino_little = indice_destino
            indice_destino = indice_destino + 1
        indice_origine = indice_origine + 1
        indice_destino = 0

    big_1x = sift2_kp2[origine_big].pt[0]
    big_1y = sift2_kp2[origine_big].pt[1]

    little_1 = sift2_kp1[idx1[idx_origine_little]]

    little_1x = little_1.pt[0]
    little_1y = little_1.pt[1]

    x_offset = big_1x - little_1x
    y_offset = big_1y - little_1y

    s_img = img1_alfa
    l_img = cv2.imread(output, 1)  #  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    y1, y2 = int(y_offset), int(y_offset + s_img.shape[0])
    x1, x2 = int(x_offset), int(x_offset + s_img.shape[1])

    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)

    alpha_s = s_img[:, :, 3] / 255
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])

    cv2.imwrite(output, l_img)

    coordinata_x = x1 + (s_img.shape[0] / 2)
    coordinata_y = y1 + (s_img.shape[1] / 2)
    angolo_di_rotazione = rotazione

    return coordinata_x, coordinata_y, angolo_di_rotazione


testo = ""

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--dbfld', dest='data_folder', type=str, default='../data/DAFNE_DB2/',
                    help='The dataset folder.')
parser.add_argument('-l', '--list', dest="fresco_folders", nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('-o', '--outfolder', dest='output_folder', type=str, default='../SOLUTION/')

args = parser.parse_args()

data_folder = args.data_folder
fresco_folders = args.fresco_folders;
output_folder = args.output_folder;
try:
    os.mkdir(output_folder)
except:
    print("Folder already exists")

# data_folder = "../data/DAFNE_DB2/"
# folder = ["01_Domenichino_Virgin-and-unicorn", "02_Zuccari_Cardinal-hat", "03_Andrea-di-Bonaiuto_Via-Veritas"]
testo = ""

for fresco_fld in fresco_folders:
    if not (fresco_fld.startswith('.')):  # in case you are working on OsX, this avoids to include .DS_Store
        # print(fresco_fld)

        img_number = int(fresco_fld[0:2])
        img_name = fresco_fld[3:]  # cut the number in front of the folder name
        img_path = data_folder + fresco_fld + "/" + img_name + ".jpg"  # path of the RGB whole image

        frag_folder = data_folder + fresco_fld + "/frag_eroded/"  # fragments path

        whole_img = cv2.imread(img_path, 0)  # this contains the whole img in GrayScale
        output = output_folder + '/RECONSTRUCTED_fresco_' + str(img_name) + '.png'
        cv2.imwrite(output, whole_img)

        max = 0
        for frag in os.listdir(frag_folder):
            frag_no = int(frag[12:-4])  # frag_no contains the fragment number
            if (frag_no > max):
                max = frag_no

        for index_folder in range(0, max + 1):

            try:

                image_input = frag_folder + "frag_eroded_" + str(index_folder) + '.png'
                # print(image_input)
                ax, ay, arot = ricomponi(image_input, img_path, output)

                elapsed_1 = time.time() - t
                t = elapsed_1

                # print(index_folder, " : ---- : ", t)
                newline = str(index_folder) + ", " + str(ax) + ", " + str(ay) + ", " + str(arot)
                testo = testo + "\n" + newline
                print(newline)

            except:
                print("fail: ", index_folder)

        testo_out = output_folder + '/fragment_' + str(img_number) + '.txt'
        out_file = open(testo_out, "w")
        out_file.write(testo)
        out_file.close()

elapsed = time.time() - t

print(elapsed)