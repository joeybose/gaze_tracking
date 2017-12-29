import numpy as np
import cv2
#from matplotlib import pyplot as plt

coeffs = np.array([.5, .5])
irisRadScale = .16
radii = np.array([0.6643555776, 1.346745904, 5])

strel = []
for i in np.arange(1,30):
    strel.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i)))

cvBuffer = np.zeros(750000,dtype=np.uint8)
resBuffer = np.zeros(3,dtype=np.float64)

counter = 0

def processEye():

    global coeffs, irisRadScale, radii, cvBuffer, resBuffer, counter

    nR, nC, dist = int(resBuffer[0]), int(resBuffer[1]), int(resBuffer[2])
    if nR < 5 or nC < 5:
        resBuffer[0], resBuffer[1], resBuffer[2] = 0, 0, 0
        return

    try:

        #scale = 50. / oC
        im = np.reshape(cvBuffer[0:nR*nC*3], (nR,nC,3))
        #im = cv2.resize(img, (0,0), fx=scale, fy=scale)
        #nR,nC = im.shape[0],im.shape[1]

        if dist < 0:
            irisRad = nC * irisRadScale
        else:
            irisRad = dist * 0.088867

        # --- Convert to YCrCb ---
        #yCrCb = cv2.cvtColor(im,cv2.COLOR_RGB2YCR_CB) / 255.
        yCrCb = cv2.cvtColor(im,cv2.COLOR_BGR2YCR_CB) / 255.
        (Y,cr,cb) = yCrCb[:,:,0], yCrCb[:,:,1], yCrCb[:,:,2]

        Ymin = np.amin(Y)
        Y = (Y - Ymin) / (np.amax(Y) - Ymin)
        cr = np.maximum(.05,(cr - .2)) / (.8 - .2)
        cb = np.maximum(.05,(cb - .2)) / (.8 - .2)
        #  --- Morphology and scaling ---
        # Apply morphological operators, if necessary, and scale by intensity component Y
        eyeMap = (cb**2 + (1 - cr)**2 + cb / cr) / 3

        if True: #eyeMapMorph.flag
            b1Diam = int(np.ceil(2 * irisRad * coeffs[0]))
            b2Diam = int(np.ceil(2 * irisRad * coeffs[1]))

            #strel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(b1Diam,b1Diam),(0,0))
            #strel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(b2Diam,b2Diam),(0,0))

            erM = cv2.dilate(eyeMap, strel[b1Diam-1])
            erY = cv2.erode(Y, strel[b2Diam-1])

            eyeMapY =  erM / (erY + .2)
        else:
            eyeMapY = eyeMap / (Y + Y.mean())

        # --- Rescale image to [0, 1] ---
        #mapMin = np.amin(eyeMapY)
        #eyeMapY = (eyeMapY - mapMin) / (np.amax(eyeMapY) - mapMin)

        size = nR*nC

        n = np.linspace(irisRad * radii[0], irisRad * radii[1], radii[2])

        # Gradient
        [dY, dX] = np.gradient(eyeMapY)
        [u, v] = np.arange(nC), np.arange(nR)[:,np.newaxis]
        gmod = np.sqrt (dX**2 + dY**2)

        pafx = dX / (gmod + 1e-4)
        pafy = dY / (gmod + 1e-4)

        # Orientation and magnitude image
        oriProj = np.zeros((nR,nC,n.size), order='F')
        magProj = np.zeros((nR,nC,n.size), order='F')

        for nIdx in np.arange(n.size):

            pPu = u + np.round(n[nIdx] * pafx).astype(int)
            pPv = v + np.round(n[nIdx] * pafy).astype(int)
            pNu = u - np.round(n[nIdx] * pafx).astype(int)
            pNv = v - np.round(n[nIdx] * pafy).astype(int)

            iP = np.nonzero(np.logical_and.reduce((pPu >= 0, pPu < nC, pPv >= 0, pPv < nR)))
            iN = np.nonzero(np.logical_and.reduce((pNu >= 0, pNu < nC, pNv >= 0, pNv < nR)))

            indsP = np.ravel_multi_index((pPv[iP], pPu[iP]), (nR,nC), order='F')
            indsN = np.ravel_multi_index((pNv[iN], pNu[iN]), (nR,nC), order='F')

            oPr = np.reshape(np.bincount(indsP, minlength=size)
                           - np.bincount(indsN, minlength=size), (nR,nC), 'F')
            mPr = np.reshape(np.bincount(indsP, gmod[iP], size)
                           - np.bincount(indsN, gmod[iN], size), (nR,nC), 'F')

            oriProj[:,:,nIdx] = np.fabs(oPr)
            magProj[:,:,nIdx] = mPr

        S = oriProj / np.amax(oriProj) * magProj / np.amax(np.fabs(magProj))

        # Smooth with the Gaussian
        imRes = np.zeros((nR,nC), order='F')
        for i in np.arange(n.size):
            sigma = n[i] / 8
            rang = np.arange(1, max(2, 3 * np.round(sigma) + 1))
            [x, y] = rang, rang[:,np.newaxis]
            c = np.round(x.shape[0] / 2.)
            G = np.exp(-((x-c)**2 + (y-c)**2) / (2 * sigma**2))
            imRes = imRes + cv2.filter2D(S[:,:,i], -1, G, borderType = cv2.BORDER_CONSTANT)

        imMin = np.amin(imRes)
        symRes = (imRes - imMin) / (np.amax(imRes) - imMin)

        # --- Find a weighted centre ---
        #[x, y] = np.meshgrid(np.arange(nC), np.arange(nR))
        #flatIm = np.ravel(symRes)
        #sumIm = np.sum(symRes)
        #cx = flatIm.dot(np.ravel(x).T) / sumIm
        #cy = flatIm.dot(np.ravel(y).T) / sumIm

        # Take pixels above 90% intensity
        [x, y] = np.meshgrid(np.arange(nC), np.arange(nR))
        idx = np.nonzero(symRes >= .9)

        subSym = symRes[idx]
        subSum = np.sum(subSym)
        cx = subSym.dot(x[idx].T) / subSum
        cy = subSym.dot(y[idx].T) / subSum

        # --- Find the brightest point ---
        #[cy,cx] = np.unravel_index([np.argmax(symRes)],(nR,nC))

        # --- Improve the estimation by excluding the background areas ---
        d = 1
        l = np.maximum(0, np.round(cx - d * irisRad))
        t = np.maximum(0, np.round(cy - d * irisRad))
        spot = symRes[t : np.minimum(np.round(cy + d * irisRad)+1, nR),
                      l : np.minimum(np.round(cx + d * irisRad)+1, nC)]

        [x2, y2] = np.meshgrid(np.arange(spot.shape[1]), np.arange(spot.shape[0]))

        # Take pixels above 25% intensity (from experiment with MUCT)
        idx = np.nonzero(spot > .25)
        if idx[0].size == 0:
            resBuffer[0], resBuffer[1], resBuffer[2] = 0, 0, 0
            return

        subSpot = spot[idx]
        subSum = np.sum(subSpot)
        cx2 = subSpot.dot(x2[idx].T) / subSum + l
        cy2 = subSpot.dot(y2[idx].T) / subSum + t

        deviation = (np.sqrt((np.fabs(u - cx2)/nC)**2.5+(np.fabs(v - cy2)/nR)**1.25) * symRes).mean()
        confidence = np.minimum(100, 3.5/deviation)
        #print confidence

        resBuffer[0] = cx2 + .5
        resBuffer[1] = cy2 + .5
        resBuffer[2] = confidence

    except:
        resBuffer[0], resBuffer[1], resBuffer[2] = 0, 0, 0

    return


#        if counter < 40:
#            counter += 1
#        elif counter < 42:
#            plt.imshow(im[:,:,::-1], interpolation = 'bicubic')
#            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#            plt.plot(resBuffer[0],resBuffer[1], 'bo')
#            plt.figure()
#            plt.imshow(imRes, cmap='gray', interpolation = 'bicubic')
#            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#            plt.plot(resBuffer[0],resBuffer[1], 'bo')
#            plt.imshow(np.maximum(0,(symRes-.5*Y-.1))**2, cmap='gray', interpolation = 'bicubic')
#            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#            plt.plot(resBuffer[0],resBuffer[1], 'bo')
#            plt.show()
#            counter += 1


