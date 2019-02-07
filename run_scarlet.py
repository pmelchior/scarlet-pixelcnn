import numpy as np
import scarlet
import logging
logger = logging.getLogger('scarlet')
logger.setLevel(logging.DEBUG)

class PixelCNNConstraint(scarlet.Constraint):
    import proxmin.operators
    def __init__(self, pixelcnn):
        self.pixelcnn = pixelcnn

    def prox_pixelcnn(self, X, step):
        grad = self.pixelcnn(X)
        return X + step*grad

    def prox_sed(self, shape):
        return proxmin.operators.prox_plus

    def prox_morph(self, shape):
        return proxmin.operators.AlternatingProjections([
            self.prox_pixelcnn,
            proxmin.operators.prox_unity,
        ])

if __name__ == "__main__":
    
    # open file and perform detection
    img = loadFile() # B x Ny x Nx
    B = len(img)
    catalog = loadCatalog() # has keys 'y' and 'x' for coordinates
    bg_rms = 1e-3 # something like sky rms of the images

    # run scarlet
    config = scarlet.Config(source_sizes=[64], accelerated=True)
    sources = [scarlet.ExtendedSource((obj['y'],obj['x']), img, bg_rms, normalization=scarlet.Normalization.S, config=config, constraints=PixelCNNConstraint(pixelcnn)) for obj in catalog]
    blend = scarlet.Blend(sources).set_data(img, config=config).fit(100, e_rel=1e-4)

    plt.figure()
    plt.plot(blend.mse)
    plt.show()

    plt.figure()
    plt.imshow(blend.get_model()[0])
    plt.show()
