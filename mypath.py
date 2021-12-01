class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'irish':
            return '/home/paul/Documents/data/clover_reg/samples/irish'
        elif dataset == 'irish_phone':
            return '/home/paul/Documents/data/clover_reg/samples/irish_phone'
        elif dataset == 'danish':
            return '/home/paul/Documents/data/clover_reg/samples/danish'
        elif dataset == 'irish_unsup':
            return '/home/paul/Documents/data/clover_reg/samples/irish_ext'
        elif dataset == 'danish_unsup':
            return '/home/paul/Documents/data/unsuplowres/images/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        
