The structure of project folder:

|---config/
|	|---__init__.py
|	|---config.py
|---dataset/
|	|---__init__.py
|	|---pascal_voc.py
|---models/
	|---nn/
		|---__init__.py
		|---etinynet.py
		|---squeezenet.py
	|---ssd/
|---results/
|	|--log/
|	|--models/
|---utils/
|	|---__init__.py
|	|---box_utils.py


Description:
=================================
=================================
../Config

	Config.py: contains the configuration parameters of trainting, such as learning rate, running device(CPU or GPU), number of epochs, Batch size
=================================
../dataset
	pascal_voc.py: A Data loader (Input pipeline), that extract the images and there annotation to use for training. the Annotations are in pascal Voc format
=================================
../models
	../nn : Contains the backbone network
		etinynet.py: contains the feature extraction architecture of EtinyNet
		squeezenet.py: contains the feature extraction architecture of SqueezeNet1.1
	../ssd: contains the full network architecture 
=================================
../utils
	box_utils.py: generate anchore boxes depending on the size of input image, it also contains functions that convert [x_min,y_min,x_max,y_min] boxes to [x_center,y_center,width,height]