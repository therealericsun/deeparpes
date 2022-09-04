# define a loading function that unpacks the data
# reshape automatically formats the angle data betewen -15 - 15 degrees

from deeparpes.main.structures import *

def load_text(path, noise=False, factor =1., squish = True, squishsize = (28,28)):
  KE_min = -0.8
  KE_max = 0.4
  Theta_min = -3.14159
  Theta_max = 3.14159
  xmin = 0
  xmax = 5
  ymin = 0
  ymax = 30
  t_i = time.time()
  print('Beginning to load data...')
  files = os.listdir(path)
  os.chdir(path)
  Count_NP = []
  preprocess = lambda x: x/x.max()
  for mat in files:
    arr = np.loadtxt(mat, skiprows=1)
    if noise:
      arr = preprocess(arr) + factor * np.random.normal(loc=0.0, scale=1.0, size=arr.shape) 
      arr = np.clip(arr, 0., 1.)
    if squish:
      arr = cv2.resize(arr, squishsize)
    Count_NP.append(arr)
    
  Count_NP = np.moveaxis(np.reshape(np.array(Count_NP), (5,30,squishsize[0],squishsize[1])), [0,1,2,3], [1,0, 3, 2])
  print(f'Data from {path} has been loaded, took {round(time.time()-t_i, 2)} seconds.')
  return ARPES_data(Count_NP, xmin, xmax, ymin, ymax, KE_min, KE_max, \
               Theta_min, Theta_max)


def load_data(path, reshape_ang=True, reshape_KE= False, KE = (40, 50)): 
  
  print('Beginning to load data...')
  t_i = time.time() # timer
  hf = h5py.File(path, 'r') # read file
  Data = hf['Data'] # Axis: KE, Slit Angle, X, Y
  Count = Data['Count']
  Count_NP = np.moveaxis(np.array(Count), [0,1], [-2, -1]) # Moved to X, Y, KE, Angle 
  Count_NP = np.flip(Count_NP, axis=2)

  xmin = 0
  xmax = Data['Axes2'].attrs.__getitem__('Count')*Data['Axes2'].attrs.__getitem__('Delta')
  ymin = 0
  ymax = Data['Axes3'].attrs.__getitem__('Count')*Data['Axes3'].attrs.__getitem__('Delta')
  KE_min = Data['Axes0'].attrs.__getitem__('Minimum')
  KE_max = Data['Axes0'].attrs.__getitem__('Maximum')
  Theta_min = Data['Axes1'].attrs.__getitem__('Minimum')
  Theta_max = Data['Axes1'].attrs.__getitem__('Maximum')
  if reshape_ang:
    if Theta_max-Theta_min <= 30:
      print("Cannot reshape, current angle bounds are too small.")
      return None
    print("Reshaping angle bounds...")
    range_0 = Count_NP.shape[3]
    range = int(range_0 *30/(Theta_max-Theta_min)) # number of samples for -15 to 15 degree arc
    Count_NP = Count_NP[:,:,:, (range_0-range)//2:(range_0-range)//2+range]
    Theta_min = -15
    Theta_max = 15
  if reshape_KE:
    print("Reshaping energy bounds...")
    range_0 = Count_NP.shape[2]
    print(KE_max, KE_min)
    print(range_0)
    range = int((KE_max-KE[0])*range_0/(KE_max-KE_min))
    range2 = int((KE_max-KE[1])*range_0/(KE_max-KE_min))
    print(range)
    Count_NP = Count_NP[:,:,range2 : range, :]
    KE_min = KE[0]
    KE_max = KE[1]
  print(f'Data from {path} has been loaded, took {round(time.time()-t_i, 2)} seconds.')

  return ARPES_data(Count_NP, xmin, xmax, ymin, ymax, KE_min, KE_max, \
               Theta_min, Theta_max) # THIS PART NEEDS REWORK TO ACTUALLY READ FROM FILE, MANUAL INPUT FOR NOW
