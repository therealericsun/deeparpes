from numpy.core.numeric import count_nonzero
import numpy as np
import h5py
import os
from math import log, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors, cm, offsetbox
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.axes_grid1 import ImageGrid
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from itertools import permutations
import numba
import random
import cv2

from deeparpes.autoencoder.data_generator import *
from deeparpes.autoencoder.noise_generator import *

@numba.jit(nopython=True)
def sum_all(x_s, y_s, data):
  clean = np.zeros((x_s, y_s))
  for x in range(x_s):
    for y in range(y_s):
      clean[x,y] = np.sum(data[x,y])
  return clean

def img_is_color(img):
    if len(img.shape) == 3:
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True
    return False


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=False, num_cols=3, figsize=(15, 9), title_fontsize=20):
    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)
        list_axes[i].axis("off")

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()




# creats a class of obejcts after applying K-means/PCA
class Clustered:
  def __init__(self, ARPES, fitted_model, raw_cords, dtype, percents): # ARPES is the ARPES_data object passed through
    self.ARPES = ARPES
    self.model = fitted_model
    self.labels = fitted_model.labels_
    self.centers = fitted_model.cluster_centers_
    self.cords = raw_cords
    self.dtype = dtype
    self.percents = percents

  def show(self, smooth = False, sampling = 1000, n_neighbors = 5):
    if smooth:
      values = self.ARPES.knn(self.labels, sampling, n_neighbors) 
    else:
      values = np.reshape(self.labels, (self.ARPES.x_s, self.ARPES.y_s)).T
    ARPES_data.plot_graph(values,'X (mm)', 'Y (mm)', \
                      [self.ARPES.xmin, self.ARPES.xmax, self.ARPES.ymin, self.ARPES.ymax], interpolation= "none", \
                      cmap = 'YlOrRd', title = f'K-means clustering for {self.dtype}', \
                      pad = 15, cbar = False)


  def show_centers(self, encode = False, enhance = False):
    print("Finding representative points...")
    closest, _ = pairwise_distances_argmin_min(self.centers, self.cords)
    print("Plotting...")
    num_centers = len(self.centers)
    rows, cols = num_centers//3+1, 3
    fig = plt.figure(figsize=(10, 6))
    points = [(i%self.ARPES.x_s, i//self.ARPES.x_s) for i in closest]
    if encode:
      values = [cv2.resize(ARPES_data.get_decoding(self.ARPES.data[p[0], p[1]], enhance), (256,256)) for p in points]
    else:
      values = [cv2.resize(self.ARPES.data[p[0], p[1]],  (256,256)) for p in points]
    titles = [f"Cluster {c}" for c in range(len(values))]
    show_image_list(values, titles, ["magma"]*len(titles))


  def show_double_cluster(self, cluster, enhance = False, line = True, epsilon = 0.4):
    points = []
    for n in range(len(self.labels)):
      if self.labels[n] == cluster:
        points.append([n%self.ARPES.x_s, n//self.ARPES.x_s])
    
    points = random.sample(points, 5)
    fig = plt.figure(figsize=(6, 10))
    fig.suptitle(f'Cluster {cluster}', fontsize=20)
    fig.subplots_adjust(top=1.1)

    grid = ImageGrid(fig, 111, 
                 nrows_ncols=(5, 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes
                 )
    
    for p in range(len(grid)//2):
      values = cv2.resize(self.ARPES.data[points[p][0], points[p][1]], (256,256))
      arped = ARPES_data.get_decoding(values, enhance)
      grid[2*p].imshow(arped, cmap = 'magma')
      if line:
        fermi = get_fermi(arped, epsilon)
        grid[2*p].axhline(y=fermi, color='red', alpha=0.5, linewidth=2)
        grid[2*p].axhline(y=fermi+43, color='green', alpha=0.3, linewidth=10)
      grid[2*p+1].imshow(values, cmap = 'magma')
      grid[2*p+1].grid(False)
      grid[2*p+1].axis('off')
      grid[2*p].grid(False)
      grid[2*p].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()
    plt.clf()

  def show_cluster(self, cluster, encode = False, enhance = False):
    print(f"Getting cluster {cluster}...")
    if cluster > len(self.centers)-1:
      print("Cluster is out of range.")
      return None
    points = []
    for n in range(len(self.labels)):
      if self.labels[n] == cluster:
        points.append([n%self.ARPES.x_s, n//self.ARPES.x_s])
    
    points = random.sample(points, 15)
    fig = plt.figure(figsize=(6, 10))
    fig.suptitle(f'Cluster {cluster}', fontsize=12)
    fig.subplots_adjust(top=0.8)

    grid = ImageGrid(fig, 111, 
                 nrows_ncols=(5, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes
                 )
    
    for p in range(len(grid)):
      if p < len(points):
        values = cv2.resize(self.ARPES.data[points[p][0], points[p][1]], (256,256))
        if encode:
          grid[p].imshow(ARPES_data.get_decoding(values, enhance), cmap = 'magma')
        else: 
          grid[p].imshow(values, cmap = 'magma')

      grid[p].grid(False)
      grid[p].axis('off')

    fig.tight_layout()
    plt.show()
    plt.clf()



  def show_distribution(self, cmap='YlOrRd'):
    scatter_x, scatter_y = self.cords[:,:2].T
    percents = self.percents
    group = self.labels

    clr_range = np.linspace(0, 1, group.max()+1)

    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c = cm.get_cmap(cmap)(clr_range[g]),\
                   edgecolors='black', label = f'Cluster {g}', s = 100)
    ax.legend(frameon=True)
    print(f"Accounting for {sum(percents[:2])} of varience.")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2") 
    plt.show()

    

  def show_3d(self, cmap='YlOrRd', show_cent=False):
    scatter_x, scatter_y, scatter_z = self.cords[:,:3].T
    percents = self.percents
    group = self.labels

    clr_range = np.linspace(0, 1, group.max()+1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')


    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c = cm.get_cmap(cmap)(clr_range[g]),\
                   edgecolors='black', label = f'Cluster {g}', s = 100)
        
    ax.legend(frameon=True)
    ax.set_xlabel("Principal Component 1", labelpad = 10.)
    ax.set_ylabel("Principal Component 2", labelpad = 10.)
    ax.set_zlabel("Principal Component 3", labelpad = 10.)


    if show_cent:
      ax2 = fig.add_subplot(111,frame_on=False) 
      ax2.axis("off")
      ax2.axis([0,1,0,1])

      # below solution from ImportanceOfBeingErnest on Github
      def proj(X, ax1, ax2):
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax1.get_proj())
        return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))
      
      def image(ax,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=0.6)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(0., -80.),
                            xycoords='data', boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle='-|>', color='black'))
        ax.add_artist(ab)

      norm = lambda x : x/x.max()

      closest, _ = pairwise_distances_argmin_min(self.centers, self.cords)
      points = [(i%self.ARPES.x_s, i//self.ARPES.x_s) for i in closest]
      images = np.array([cv2.resize(self.ARPES.data[p[0], p[1]],  (64,64)) for p in points])
      values = [cm.magma(norm(i)) for i in images] # all the images

      xs, ys, zs = np.array([self.cords[:, :3][c] for c in closest]).T


      for s in zip(xs,ys,zs, values):
        x,y = proj(s[:3], ax, ax2)
        image(ax2,s[3],[x,y])

    print(f"Accounting for {sum(percents[:3])} of varience.")
    plt.show()


  def get_accuracy(self, gt):
    assert len(gt.shape) == 2
    vals = []
    tests = list(permutations(range(int(gt.max())+1)))
    print(tests)
    labelswitch = np.zeros(gt.shape)
    x, y = gt.shape
    for label in tests:
      for i in range(x):
        for j in range(y):
          labelswitch[i,j] = label[int(np.reshape(self.labels, (30, 30), order = 'F')[i,j])]
      vals.append(np.count_nonzero(gt-labelswitch))
    return round(1-min(vals)/(x*y), 3)



  def PCA_on_autoencoder(self, bad_label = 0):
    pca_model = PCA(0.95)
    pca_model2 = PCA(0.95)
    cleaned_cords = []
    raw_cords = []
    for cord in range(len(self.labels)):
      if self.labels[cord] != bad_label:
        cleaned_cords.append(self.cords[cord])
        raw_cords.append(1)
      else:
        raw_cords.append(0)
    cc = np.array(cleaned_cords)
    t = ((pca_model2.fit_transform(cc)).T[0]).T
    final = []
    counter = 0
    for i in range(len(raw_cords)):
      if raw_cords[i] == 0:
        final.append(0)
      else:
        final.append(t[counter])
        counter += 1
    return np.array(final)
    
  def zerocount(self, bad_label, epsilon = 0.5):
    x_s = self.ARPES.x_s
    y_s = self.ARPES.y_s
    map = np.zeros((x_s,y_s))
    lab = np.reshape(self.labels, (x_s, y_s))
    for x in range(x_s):
      for y in range(y_s):
        if lab[x,y] != bad_label:
          map[x,y] = get_fermi(ARPES_data.get_decoding(self.ARPES.data[y,x], True), epsilon)
        else:
          map[x,y] = 0
    return map


# create custom class for ARPES data
class ARPES_data:
  def __init__(self, data, xmin, xmax, ymin, ymax, KE_min, KE_max, \
               Theta_min, Theta_max):
    self.data = data
    self.x_s, self.y_s, self.Ke_s, self.Ang_s = data.shape # also loads shape initially
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.KE_min = KE_min
    self.KE_max = KE_max
    self.Theta_min = Theta_min
    self.Theta_max = Theta_max


  # enables the grabbing of data values by subscripting
  def __getitem__(self, key): 
    return self.data[key]


  # makes default representation the raw data in matrix form
  def __repr__(self): 
    return self.data


  # plot the data using this wrapper function (notes: everything follows ply, Autosqure automatically scales the thing)
  # specifaiclly used for matricies
  @staticmethod 
  def plot_graph(values, xlabel, ylabel, ext, xbins=6, ybins=6, \
                 interpolation = "lanczos", aspect=1, cmap = "magma",\
                 title = "", pad = 0, Autosquare = False, cbar = True, show = True,
                 axis = True): 
    fig = plt.figure(figsize=(6, 6))
    print("Drawing graph...")
    if Autosquare: # automatically make thing square
      aspect = (ext[1]-ext[0])/(ext[3]-ext[2])
    
    fig, ax = plt.subplots()
    plt.imshow(values, extent = ext, interpolation = interpolation, \
               aspect=aspect, cmap = cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    plt.title(title, pad=pad) 
    plt.grid(False)
    if not axis:
      plt.axis('off')
    if cbar:
      plt.colorbar()
    else: # create custom legend
      elements = []
      n_clusters = np.amax(values)+1
      clr_range = np.linspace(0, 1, n_clusters)
      for cluster in range(n_clusters):
        elements.append(Patch(label = f"Cluster {cluster}", facecolor = cm.get_cmap(cmap)(clr_range[cluster])))
      ax.legend(handles = elements, loc = 'lower right', bbox_to_anchor=(1.3, 0.4), frameon=True)
    if show:
      plt.show()
      plt.clf()

  # transform raw data to processable cv2 img (1-d array form when flattened)
  def clust_transform(self, img, dim =(256,256)):
    img = cv2.resize(img, dim) # create greyscale cv2 img
    img = np.reshape(img, (1,-1))[0] # create 1xn array
    return img

  # creates graph of the object integrated over all angles and energies
  def show(self, Interpolation="none"): 
    t_0 = time.time()
    print("Formating data for graph...") # sum over all angles and values
    self.plot_graph(sum_all(self.x_s, self.y_s, self.data), 'X (mm)', 'Y (mm)', [self.xmin, self.xmax, self.ymin, self.ymax], interpolation = Interpolation)

  @staticmethod
  def get_encoding(img, enhance = False):
    if 'autoencoder_fitted' not in globals():
      print("Must fit autoencoder first.")
      return None
    encoder = keras.Model(input_img, encoded)
    val = np.array([norm_ln(img)]) if enhance else np.array([norm(img)])
    return np.reshape(encoder.predict(val), (64,32))

  @staticmethod
  def get_batch(imgs, enhance = False):
    if 'autoencoder_fitted' not in globals():
      print("Must fit autoencoder first.")
      return None
    encoder = keras.Model(input_img, encoded)
    val = norm_ln_batch(imgs) if enhance else norm_batch(imgs)
    val = np.array([np.reshape(n, (256,256, 1)) for n in val])
    return encoder.predict_on_batch(val)


  @staticmethod
  def get_decoding(img, enhance = False):
    if 'autoencoder_fitted' not in globals():
      print("Must fit autoencoder first.")
      return None
    val = np.array([norm_ln(img)]) if enhance else np.array([norm(img)])
    return np.reshape(autoencoder.predict(val), (256,256))


  # shows deep-learning embedding
  def show_encoding(self, l_x, l_y):
    print(f"Formating encoding for point ({l_x},{l_y})...")
    graph = cv2.resize(self.get_encoding(self.data[l_x,l_y]), (256,256))
    self.plot_graph(graph,'', '', \
                    [0, 1, 0, 1], aspect = 1, \
                    title = f"Cords: ({l_x},{l_y})", pad = 15, axis = False,
                    cmap = 'gray',  interpolation = "none")

  def show_decoding(self, l_x, l_y, enhance = False):
    print(f"Formating decoding for point ({l_x},{l_y})...")
    decode = self.get_decoding(self.data[l_x,l_y], enhance = True)
    graph = cv2.resize(decode, (256,256))*255
    self.plot_graph(graph,'', '', \
                    [0, 1, 0, 1], aspect = 1, \
                    title = f"Cords: ({l_x},{l_y})", pad = 15, axis = False,
                    cmap = 'gray',  interpolation = "none")

  # plots angle-KE data at specific l_x l_y cord
  def show_point(self, l_x, l_y, mode = 'default'): 
    print(f"Formating data for point ({l_x},{l_y})...")
    graph = self.data[l_x, l_y] # look at the l_x, l_y point of the data

    if mode=='ln': # ln compression if needed
      vlog = lambda i : log(i+1)
      mat_log = np.vectorize(vlog)
      graph = mat_log(graph)
    elif mode=='sqrt': # quadretic compression if needed
      vlog = lambda i : sqrt(i)
      mat_log = np.vectorize(vlog)
      graph = mat_log(graph)
    elif mode!='default': # check cases
      print('Mode must be "default", "ln", or "sqrt". Leave blank for default.')
      return None
  
    self.plot_graph(graph,'Theta (deg)', 'KE (eV)', \
                    [self.Theta_min, self.Theta_max, self.KE_min, self.KE_max], aspect = 40, \
                    title = f"Cords: ({l_x},{l_y})", pad = 15, Autosquare = True)


  # integrate along energy (returns angle-point-intensity graph)
  def integrate_energy(self): 
    print("Integrating energy...")
    graph = np.zeros((self.x_s*self.y_s, self.Ang_s))
    for x in range(self.x_s):
      for y in range(self.y_s):
        graph[self.x_s*y+x] = self.data[x,y].sum(0) # replace each row with compressed data row
    return graph


  # integrate along angle (returns energy-point-intensity graph)
  def integrate_angle(self): 
    print("Integrating angle...")
    graph = np.zeros((self.x_s*self.y_s, self.Ke_s))
    for x in range(self.x_s):
      for y in range(self.y_s):
        graph[self.x_s*y+x] = self.data[x,y].sum(1) # replace each row with compressed data row
    return graph

  # autoencoder over surface 
  def autoencode(self, enhance=False):
    print("Autoencoding data...")
    flat = np.concatenate(np.swapaxes(self.data, 0, 1))
    graph = np.reshape(self.get_batch(flat, enhance=enhance), (self.x_s * self.y_s, 2048))
    return graph

  # graph the energy data in terms of the indivisual points
  def show_energy(self): 
    print("Formatting data for energy...")
    graph = self.integrate_energy()
    self.plot_graph(graph, "Angle (deg)", "Point", \
                    ext = [self.Theta_min, self.Theta_max, 0, self.x_s*self.y_s], \
                    Autosquare = True, interpolation = "none")


  # graph the angle data in terms of the indiviusal points
  def show_angle(self): 
    print("Formatting data for angle...")
    graph = self.integrate_angle()
    self.plot_graph(graph, "KE (eV)", "Point", \
                    ext = [self.KE_min, self.KE_max, 0, self.x_s*self.y_s], \
                    Autosquare = True, interpolation= "none")
  
  # graph autoencoder distributions
  def show_autoencode(self, enhance=False):
    print("Formatting data for autoencoding...")
    graph = self.autoencode(enhance)
    self.plot_graph(graph, "", "Point", \
                    ext = [0, 2048, 0, self.x_s*self.y_s], \
                    Autosquare = True, interpolation= "none", axis = False)


  # perform PCA dimentionality reduction
  def PCA(self, dim = (256,256), n_components = 0.95):
    t_i = time.time()
    print("Beginning PCA...")
    model = PCA(n_components)
    values = np.zeros((self.x_s*self.y_s, dim[0]*dim[1]))
    for x in range(self.x_s):
      for y in range(self.y_s):
        values[x+y*self.x_s] = self.clust_transform(self.data[x,y], dim = dim)
    model.fit(values)
    print(f"PCA reduction complete, took {round(time.time()-t_i, 2)} seconds.")
    return model.transform(values), model.explained_variance_ratio_


  # kmeans
  def kmeans(self, dtype, n_clusters, dim = (256,256), n_components = 0.95, enhance = False): 
    if dtype not in ['angle', 'energy', 'PCA', 'autoencoder']:
      print("type must be 'angle', 'energy', 'PCA', or 'autoencoder'.")
      return None
    
    km = KMeans(n_clusters, n_init = 10, random_state = 100) # create kmeans
    
    if dtype == 'PCA':
      target, percents = self.PCA(dim, n_components)
    elif dtype == 'autoencoder':
      target = self.autoencode(enhance)
      percents = None
    else:
      target = self.integrate_angle() if dtype == 'angle' else self.integrate_energy()
      percents = None

    print(f"Beginning kmeans clustering, type is {dtype}...")
    t_i = time.time()
    vals = km.fit(target)
    print(f"Kmeans clustering complete, took {round(time.time()-t_i, 2)} seconds.")
    return Clustered(self, vals, target, dtype, percents)


  # knn-based smoothing
  def knn(self, results, sampling, n_neighbors): 
    print("Begninning knn smoothing...")
    t_i = time.time()
    c_product = lambda x,y : np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]) # cartesion product lambda function
    x_range, y_range = np.linspace(self.xmin, self.xmax, self.x_s), np.linspace(self.ymin, self.ymax, self.y_s)
    dataspace = c_product(x_range, y_range) # create grid of all known values
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(dataspace, results) # fit KNN model: first term is location, second term is type
    xx, yy = np.meshgrid(np.linspace(self.xmin, self.xmax, sampling),\
                          np.linspace(self.ymin, self.ymax, sampling))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # start sampling the created space using the KNN alg
    Z = Z.reshape(xx.shape)
    print(f"Knn smoothing complete, took {round(time.time()-t_i,2)} seconds.")
    return Z.T


  # wrapper to print object
  def show_kmeans(self, dtype, n_clusters, smooth=False, sampling=2000, n_neighbors = 5, Autosquare = True, enhance = False):
    results = self.kmeans(dtype, n_clusters, enhance = enhance).labels
    if smooth:
      graph = self.knn(results, sampling, n_neighbors) # extrapolate in-between values using knn alg. above
    else:
      graph = np.reshape(results, (self.x_s, self.y_s), order='F') # return linearly ordered data to 2-d graph
    
    self.plot_graph(graph,'X (mm)', 'Y (mm)', \
                      [self.xmin, self.xmax, self.ymin, self.ymax], interpolation= "none", \
                      cmap = 'YlOrRd', title = f'K-means clustering for {dtype}', \
                      pad = 15, cbar = False, Autosquare = Autosquare)

  # visialiezer for the elbow method via yellowbrick
  def elbow(self,dtype,range=(3,8), dim=(256,256), n_components=0.95, enhance=False): 
    print(f"Visualising elbow method for '{dtype}'...")
    model = KMeans()
    if dtype not in ['angle', 'energy', 'PCA', 'autoencoder']:
      print("type must be 'angle', 'energy', 'autoencoder', or 'PCA'")
      return None

    if dtype == 'PCA':
      target, _ = self.PCA(dim, n_components)
    elif dtype == 'autoencoder':
      target = self.autoencode(enhance)
    else:
      target = self.integrate_angle() if dtype == 'angle' else self.integrate_energy()
    visualizer = KElbowVisualizer(model, k=range)
    visualizer.fit(target)
    visualizer.show() 


  def silhouette(self,dtype,range=(3,8), dim=(256,256), n_components=0.95, enhance = False): 
    print(f"Visualising elbow method for '{dtype}'...")
    model = KMeans()
    if dtype not in ['angle', 'energy', 'PCA', 'autoencoder']:
      print("type must be 'angle', 'energy', or 'PCA'")
      return None

    if dtype == 'PCA':
      target, _ = self.PCA(dim, n_components)
    elif dtype == 'autoencoder':
      target = self.autoencode(enhance)
    else:
      target = self.integrate_angle() if dtype == 'angle' else self.integrate_energy()
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(target)
    visualizer.show() 
