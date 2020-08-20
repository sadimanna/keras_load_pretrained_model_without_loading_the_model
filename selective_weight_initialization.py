def selective_load_weights(new_model, old_model_file_path):
    # LOADING THE FILE
    old_model_file = h5py.File(old_model_file_path,'r')
    model_weights = old_model_file['model_weights']

    # THE LAYER NAMES IN THE OLD MODEL IS AVAILABLE AS KEYS
    old_model_layers = list(model_weights.keys())

    # THE NAMES OF THE LAYERS IN THE NEW MODEL
    new_model_layers = [l.name for l in new_model.layers]
    #print(old_model_layers)
    
    # THE ATTRIBUTES WHOSE VALUES NEEDS TO BE COPIED
    key_dict = {'conv2d':['kernel','bias'],
                'dense':['kernel','bias'],
                'batch_normalization':['gamma','beta','moving_mean','moving_variance']}
    
    # DIFFERENT TYPES OF LAYER PRESENT IN THE OLD MODEL
    types_of_layers = list(set([l.rstrip('0123456789_') for l in old_model_layers]))
    #print(types_of_layers)

    # GROUP THE LAYERS ACCORDING TO THEIR TYPE
    old_layers_dict = {}
    for tol in types_of_layers:
        old_layers_dict[tol] = [l for l in old_model_layers if tol in l]
    #print(old_layers_dict)
    
    # GROUP THE LAYERS ACCORDING TO THEIR TYPE
    # TAKE THE HIGHESTINDEX OF EACH TYPE OF LAYER (CAN ALSO BE DONE BY TAKING THE LENGTH OF THE LIST)
    new_layers_dict = {}
    for tol in types_of_layers:
        new_layers_dict[tol] = [l for l in new_model_layers if tol in l]
        # The last layer to be added will have the highest index
        if new_layers_dict[tol] != []:
            highestindex = new_layers_dict[tol][-1].split('_')[-1]
            if not highestindex.isdigit():
                highestindex = 0
            else:
                highestindex = int(highestindex)
        else:
            highestindex = None
        
        # RENAME THE NAMES OF THE LAYERS TO SORT THEM IN ORDER OF THEIR NUMBERING
        # OTHERWISE 'conv2d_2' comes after 'conv2d_13' after sorting the string, 
        # SO converting 'conv2d_2' to 'conv2d_02'
        if highestindex !=None:
            lenindex = len(str(highestindex))
            for i in range(highestindex):
                nld = new_layers_dict[tol][i].split('_')
                if not nld[-1].isdigit():
                    nld.append('0'*lenindex)
                elif nld[-1].isdigit():
                    nld[-1] = (lenindex-len(nld[-1]))*'0' + nld[-1]
                new_layers_dict[tol][i] = '_'.join([n for n in nld])
                new_model
        
        new_layers_dict[tol] = sorted(new_layers_dict[tol])
        #print(new_layers_dict[tol])
    
    # INITIALIZING THE WEIGHTS
    for k in list(key_dict.keys()):
    	# TAKE A LIST FOR EACH TYPE OF LAYER WHICH HAVE WEIGHTS
        old_layers = old_layers_dict[k]
        new_layers = new_layers_dict[k]
        
        # FOR EACH LAYER IN THE NEW MODEL THAT NEEDS INTIIALIZING
        for i in range(len(new_layers)):
            print(new_layers[i],old_layers[i])

            # CONVERT THE LAYER NAMES BACK TO THEIR ORIGINAL FORM TO RETRIEVE THE LAYERS FROM THE NEW MODEL
            nld = new_layers[i].split('_')
            nld[-1] = nld[-1].lstrip('0')
            if nld[-1]=='':
                nld = nld[:-1]
            layer_name = '_'.join(nld)
            #layer = new_model.get_layer(layer_name)
            
            # GET THE GROUP FROM THE H5 FILE WHICH CONTAINS THE WEIGHTS OF THE LAYER
            param_group = model_weights[old_layers[i]][old_layers[i]]
            var_keys = key_dict[k]
            weight_list = []
            #print(new_model.get_layer(layer_name).weights)

            # FOR EACH KEY IN THE GROUP READ THE DATA FROM THE CORRESPONDING DATASETS
            # AND APPEND TO THE LIST OF WEIGHTS
            for key in var_keys:
                #print(key)
                dset = param_group[key+':0']
                shape = dset.shape
                dtype = dset.dtype
                arr = np.empty_like(np.array([]),shape=shape,dtype=dtype)
                dset.read_direct(arr)
                weight_list.append(arr)
            #print(weight_list)

            #SET THE WEIGHTS
            new_model.get_layer(layer_name).set_weights(weight_list)
    
    old_model_file.close()
    # RETURN THE INITIALIZED MODEL
    return new_model
