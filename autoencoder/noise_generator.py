from deeparpes.autoencoder.data_generator import *

h, w = 256,256
blur = lambda img : b(img, (10,10) ,cv2.BORDER_DEFAULT)
reshape = lambda img : np.reshape(img, ((len(img), h, w, 1)))
norm = lambda img: cv2.resize(img, (256, 256))/cv2.resize(img, (256, 256)).max()

def norm_batch(imgs):
  return np.array([norm(img) for img in imgs])

# GPU accerlated fast calculation
@numba.jit(nopython=True)
def _norm_ln(img):
  x,y = img.shape
  ret = np.zeros((x,y))
  for i in range(x):
    for j in range(y):
      ret[i,j] = log(img[i,j]+1)
  return ret

def norm_ln(img):
  x = cv2.resize(_norm_ln(img), (256,256))
  return x/x.max()

# GPU accerlated fast calculation
@numba.jit(nopython=True)
def _norm_ln_batch(imgs):
  n,x,y = imgs.shape
  ret = np.zeros((n,x,y))
  for t in range(n):
    ret[t] = _norm_ln(imgs[t])
  return ret

def norm_ln_batch(imgs):
  x = np.array([cv2.resize(img, (256,256)) for img in _norm_ln_batch(imgs)])
  return x/x.max()

# noise_factor changes noise level, hide_factor reduces curve visibility, b for blur
# noise factor = 0.25, hide factor = 0.25 for BSCCO
# n_f = 0.5, h_f = 1 for TaSe2
# def make_noise(imgs, noise_factor = 0.2, hide_factor = 0.8, b = True):
def make_noise(imgs, noise_factor = 0.8, hide_factor = 1., b = True):
  x = reshape(np.array(list(map(blur, imgs)))) if b else reshape(imgs)
  x = hide_factor*x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
  return np.clip(x, 0., 1.)


# @numba.jit(nopython=True)
def poisson(imgs, noise_factor = 0.4, hide_factor = 0.7):
  results = []
  # for i in range(len(imgs)):
  #   results.append(noise_factor*imgs[i] + hide_factor*np.random.poisson(imgs[i]))
  for i in range(len(imgs)):
     results.append(hide_factor*np.random.poisson(imgs[i]))
  return np.array(results)


def make_p_noise(imgs, noise_factor = 0.4, hide_factor = 0.4, b = True):
  x = poisson(imgs)
  x = reshape(np.array(list(map(blur, x)))) if b else reshape(x)
  return np.clip(x, 0., 1.)

noisy = lambda img : make_noise(make_p_noise(img, hide_factor = 0.5), b = True, noise_factor = 0.3, hide_factor=.4)
