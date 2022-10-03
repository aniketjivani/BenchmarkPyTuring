import seaborn as sns
import numpy as np
import pandas as pd
import pymc3 as pm
RANDOM_SEED = 974067
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from collections import OrderedDict
import pickle

import theano
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_compile'
import theano.tensor as T
from   theano.tensor.nlinalg import matrix_inverse


class DirichletAllocationProcess:
    def __init__(self, name, outputs):
        assert len(outputs) >= 1
        self.name = name
        self.outputs = outputs
        self.nparams = len(outputs)

    @staticmethod
    def transfer_functions(params):
        return params

    @staticmethod
    def prior(shares, concentration=None, with_stddev=None):
        if (concentration is not None and with_stddev is not None) or \
           (concentration is None and with_stddev is None):
            raise ValueError('Specify either concentration or stddev')


        # normalise
        factor = sum(shares)
        shares = np.array(shares) / factor

        if with_stddev is not None:
            i, stddev = with_stddev
            stddev /= factor
            mi = shares[i]
            limit = np.sqrt(mi * (1 - mi) / (1 + len(shares)))
            concentration = mi * (1 - mi) / stddev**2 - 1
            if not np.isfinite(concentration):
                concentration = 1e10
        else:
            concentration = len(shares) * concentration
        return concentration * shares


    def param_rv(self, pid, defs):
        if defs is None:
            defs = np.ones(self.nparams)
        assert len(defs) == self.nparams
        if len(defs) > 1:
            return pm.Dirichlet('param_{}'.format(pid), defs)
        else:
            return pm.Deterministic('param_{}'.format(pid), T.ones((1,)))


class SinkProcess:
    def __init__(self, name):
        self.name = name
        self.outputs = []
        self.nparams = 0

    @staticmethod
    def transfer_functions(params):
        return T.dvector()

    def param_rv(self, pid, defs):
        return None


dir_prior = DirichletAllocationProcess.prior
def define_processes():

    Import_Iron_Ore           = DirichletAllocationProcess('Import_Iron_Ore_Allocation',       ['Iron_Ore_Consumption'])
    Iron_Ore_Production       = DirichletAllocationProcess('Iron_Ore_Production_Allocation',   ['Export', 'Iron_Ore_Consumption'])         ###
    Iron_Ore_Consumption      = DirichletAllocationProcess('Iron_Ore_Consumption_Allocation',  ['Blast_Furnace','DRI_Production','Other'])
    Blast_Furnace             = DirichletAllocationProcess('Blast_Furnace_Allocation',         ['Loss','Pig_Iron'])

    Import_Scrap              = DirichletAllocationProcess('Import_Scrap_Allocation',          ['Scrap_Consumption'])
    Purchased_Scrap           = DirichletAllocationProcess('Purchased_Scrap_Allocation',       ['Scrap_Collected'])
    Scrap_Collected           = DirichletAllocationProcess('Scrap_Collected_Allocation',       ['Export','Scrap_Consumption'])
    Scrap_Consumption         = DirichletAllocationProcess('Scrap_Consumption_Allocation',     ['Blast_Furnace', 'Basic_Oxygen_Furnace',
                                                                                               'Electric_Arc_Furnace','Cupola', 'Other_Casting'])

    Import_Pig_Iron           = DirichletAllocationProcess('Import_Pig_Iron_Allocation',       ['Pig_Iron_Consumption'])
    Pig_Iron                  = DirichletAllocationProcess('Pig_Iron_Allocation',              ['Export','Pig_Iron_Consumption']) 
    Pig_Iron_Consumption      = DirichletAllocationProcess('Pig_Iron_Consumption_Allocation',  ['Basic_Oxygen_Furnace','Electric_Arc_Furnace',
                                                                                                'Cupola', 'Other_Casting'])
    Import_DRI                = DirichletAllocationProcess('Import_DRI_Allocation',            ['DRI_Consumption'])        
    DRI_Production            = DirichletAllocationProcess('DRI_Production_Allocation',        ['Loss', 'DRI'])
    DRI                       = DirichletAllocationProcess("DRI_Allocation",                   ['Export', 'DRI_Consumption'])
    DRI_Consumption           = DirichletAllocationProcess('DRI_Consumption_Allocation',       ['Blast_Furnace', 'Basic_Oxygen_Furnace',
                                                                                               'Electric_Arc_Furnace', 'Cupola', 'Other_Casting'])
    Basic_Oxygen_Furnace      = DirichletAllocationProcess('Basic_Oxygen_Furnace_Allocation',  ['Continuous_Casting','Loss'])  
    Electric_Arc_Furnace      = DirichletAllocationProcess('Electric_Arc_Furnace_Allocation',  ['EAF_Yield','Loss']) 
    
    EAF_Yield                 = DirichletAllocationProcess('EAF_Yield_Allocation',             ['Continuous_Casting', 'Ingot_Casting'])
    Cupola                    = DirichletAllocationProcess('Cupola_Allocation',                ['Ingot_Casting', 'Loss'])

    Continuous_Casting        = DirichletAllocationProcess('Continuous_Casting_Allocation',    ['CC_Yield', 'CC_Loss']) 
    CC_Loss                   = DirichletAllocationProcess('CC_Loss_Allocation',               ['CC_Yield', 'Loss']   )
    CC_Yield                  = DirichletAllocationProcess('CC_Yield_Allocation',              ['Hot_Strip_Mill', 'Plate_Mill', 'Rod_and_Bar_Mill', 'Section_Mill'])

    Ingot_Casting             = DirichletAllocationProcess('Ingot_Casting_Allocation',         ['IC_Yield', 'IC_Loss']) 
    IC_Loss                   = DirichletAllocationProcess('IC_Loss_Allocation',               ['IC_Yield', 'Loss']   )
    IC_Yield                  = DirichletAllocationProcess('IC_Yield_Allocation',              ['Primary_Mill']) 
    
    Other_Casting             = DirichletAllocationProcess('Other_Casting_Allocation',         ['OC_Yield', 'OC_Loss']) 
    OC_Loss                   = DirichletAllocationProcess('OC_Loss_Allocation',               ['OC_Yield', 'Loss']   )
    OC_Yield                  = DirichletAllocationProcess('IC_Yield_Allocation',              ['Steel_Product_Casting','Iron_Product_Casting'])

    Hot_Strip_Mill            = DirichletAllocationProcess('Hot_Strip_Mill_Allocation',     ['HSM_Yield', 'Scrap_Consumption'])     
    HSM_Yield                 = DirichletAllocationProcess('HSM_Yield_Allocation',          ['Cold_Rolling_Mill', 'Hot_Rolled_Sheet','Pipe_Welding_Plant'])     
    
    Cold_Rolling_Mill         = DirichletAllocationProcess('Cold_Rolling_Mill_Allocation',  ['CRM_Yield', 'Scrap_Consumption'])
    CRM_Yield                 = DirichletAllocationProcess('CRM_Yield_Allocation',          ['Cold_Rolled_Sheet', 'Galvanized_Plant', 'Tin_Mill'])                                          
    
    Rod_and_Bar_Mill          = DirichletAllocationProcess('Rod_and_Bar_Mill_Allocation',   ['RBM_Yield', 'Scrap_Consumption']) 
    RBM_Yield                 = DirichletAllocationProcess('RBM_Yield_Allocation',          ['Seamless_Tube_Plant', 'Bars', 'Reinforcing_Bars', 'Wire_and_Wire_Rods', 'Light_Section'])

    Section_Mill              = DirichletAllocationProcess('Section_Mill_Allocation',       ['SM_Yield', 'Scrap_Consumption']) 
    SM_Yield                  = DirichletAllocationProcess('SM_Yield_Allocation',           ['Rails_and_Rail_Accessories', 'Heavy_Section']) 
    
    Primary_Mill              = DirichletAllocationProcess('Primary_Mill_Allocation',       ['PM_Yield', 'Scrap_Consumption'])
    PM_Yield                  = DirichletAllocationProcess('PM_Yield_Allocation',           ['Export', 'Hot_Strip_Mill', 'Cold_Rolling_Mill', 'Plate_Mill', 'Rod_and_Bar_Mill','Section_Mill'])

    Plate_Mill                = DirichletAllocationProcess('Plate_Mill_Allocation',         ['Plates', 'Scrap_Consumption']) 
    Tin_Mill                  = DirichletAllocationProcess('Tin_Mill_Allocation',           ['Tin_Mill_Products', 'Scrap_Consumption'])      
    
    Galvanized_Plant          = DirichletAllocationProcess('Gal_Plant_Allocation',          ['Galvanized_Sheet', 'Scrap_Consumption'])     
    Pipe_Welding_Plant        = DirichletAllocationProcess('Pipe_Welding_Plant_Allocation', ['Pipe_and_Tubing',  'Scrap_Consumption'])
    Seamless_Tube_Plant       = DirichletAllocationProcess('Seamless_Tube_Allocation',      ['Pipe_and_Tubing',  'Scrap_Consumption'])

    Cold_Rolled_Sheet         = DirichletAllocationProcess('Cold_Rolled_Sheet_Allocation',   ['Export', 'Automotive',   'Machinery', 'Steel_Products', 'Scrap_Consumption'])
    Galvanized_Sheet          = DirichletAllocationProcess('Galvanized_Sheet_Allocation',    ['Export', 'Construction', 'Automotive', 'Scrap_Consumption'])                                                                          
    Tin_Mill_Products         = DirichletAllocationProcess('Tin_Mill_Products_Allocation',   ['Export', 'Automotive',   'Steel_Products', 'Scrap_Consumption'])
    Hot_Rolled_Sheet          = DirichletAllocationProcess('Hot_Rolled_Sheet_Allocation',    ['Export', 'Construction', 'Automotive', 'Machinery', 'Energy', 'Steel_Products', 'Scrap_Consumption'])

    Pipe_and_Tubing           = DirichletAllocationProcess('Pipe_and_Tubing_Allocation',     ['Export', 'Construction', 'Automotive', 'Machinery', 'Energy',   'Scrap_Consumption'])                                                              
    Plates                    = DirichletAllocationProcess('Plates_Allocation',              ['Export', 'Construction', 'Automotive', 'Machinery', 'Energy',   'Scrap_Consumption'])
    Reinforcing_Bars          = DirichletAllocationProcess('Reinforcing_Bars_Allocation',    ['Export', 'Construction', 'Scrap_Consumption'])
    Bars                      = DirichletAllocationProcess('Bars_Allocation',                ['Export', 'Construction', 'Automotive', 'Machinery', 'Energy',   'Scrap_Consumption'])
    Wire_and_Wire_Rods        = DirichletAllocationProcess('Wires_and_Wire_Rods_Allocation', ['Export', 'Construction', 'Automotive', 'Machinery', 'Energy',   'Scrap_Consumption'])

    Rails_and_Rail_Accessories = DirichletAllocationProcess('Rails_and_Rail_Accessories_Allocation',  ['Export', 'Construction', 'Machinery', 'Scrap_Consumption'])
    Light_Section              = DirichletAllocationProcess('Light_Sections_Allocation',              ['Export', 'Construction', 'Automotive', 'Scrap_Consumption'])  
    Heavy_Section              = DirichletAllocationProcess('Heavy_Sections_Allocation',              ['Export', 'Construction', 'Scrap_Consumption'])    
    Steel_Product_Casting      = DirichletAllocationProcess('Steel_Casting_Allocation',               [ 'Construction', 'Automotive','Machinery', 'Energy'])                                                                                                                                            
    Iron_Product_Casting       = DirichletAllocationProcess('Iron_Casting_Allocation',                [ 'Construction', 'Automotive','Machinery', 'Energy'])

    Ingot_Import               = DirichletAllocationProcess('Import_of_Ingot',  ['Primary_Mill'])  

    Intermediate_Product_Import = DirichletAllocationProcess('Intermediate_Product_Import_Allocation', ['Cold_Rolled_Sheet', 'Galvanized_Sheet', 'Tin_Mill_Products', 
                                                             'Hot_Rolled_Sheet', 'Pipe_and_Tubing', 'Plates', 'Reinforcing_Bars','Bars',
                                                             'Wire_and_Wire_Rods', 'Rails_and_Rail_Accessories', 'Light_Section',
                                                             'Heavy_Section', 'Steel_Product_Casting', 'Iron_Product_Casting'])

    Export         = SinkProcess('Sink1') 
    Loss           = SinkProcess('Sink2')
    Other          = SinkProcess('Sink3')
    Construction   = SinkProcess('Sink4')
    Automotive     = SinkProcess('Sink5')
    Machinery      = SinkProcess('Sink6')
    Energy         = SinkProcess('Sink7')
    Steel_Products = SinkProcess('Sink8')


    return OrderedDict((pid, process) 
        for pid, process in sorted(locals().items())) 
processes = define_processes()


input_observations = [               
    [   
    (['Import_DRI'],                  2.47), 
    (['Import_Iron_Ore'],             5.16),
    (['Import_Pig_Iron'],             4.27), 
    (['Import_Scrap'],                3.72), 
    (['Ingot_Import'],                6.94 ),
    (['Intermediate_Product_Import'], 23.46),
    (['Iron_Ore_Production'],         54.7), 
    (['Purchased_Scrap'],             70.98),
]]


input_defs = [ #Here all the inputs sources are associated with observations, might not hold for other studys
    { 
    'Import_DRI':                  2.47,
    'Import_Iron_Ore':             5.16, 
    'Import_Pig_Iron':             4.27, 
    'Import_Scrap':                3.72,  
    'Ingot_Import':                6.94,
    'Intermediate_Product_Import': 23.46,
    'Iron_Ore_Production':          54.7,      
    'Purchased_Scrap':             70.98,
     },  ]



param_defs0 = {
    'Iron_Ore_Production': dir_prior(np.array([1.39, 6.82])/sum(np.array([1.39, 6.82]))*100,                 with_stddev = [0, 12]),
    'Iron_Ore_Consumption': dir_prior(np.array([17.25, 1.73, 0.60])/sum(np.array([17.25, 1.73, 0.60]))*100,  with_stddev = [0, 7]),

    'DRI_Production'      : dir_prior(np.array([0.41, 1.73])/sum(np.array([0.41, 1.73]))*100, with_stddev = [0, 22]),
    'DRI'                 : dir_prior(np.array([0.17, 0.73])/sum(np.array([0.17, 0.73]))*100, with_stddev = [0, 28] ),
    'DRI_Consumption'     : dir_prior(np.array([1.6, 1.09, 21.47, 0.1, 0.77])/sum(np.array([1.6, 1.09, 21.47, 0.1, 0.77]))*100, with_stddev=[0, 5]),

    'Scrap_Collected'     : dir_prior(np.array([1.16, 4.18])/sum(np.array([1.16, 4.18]))*100, with_stddev = [0, 16]),
    'Scrap_Consumption'   : dir_prior(np.array([ 0.90, 1.62, 5.82, 0.59, 0.1 ])/sum(np.array([0.90, 1.62, 5.82, 0.59, 0.1 ]))*100, with_stddev=[0, 9]),
    
    'Blast_Furnace'       : dir_prior(np.array([2.42, 2.13])/sum(np.array([2.42, 2.13]))*100, with_stddev = [0, 21]),
  
    'Pig_Iron'            : dir_prior(np.array([0.79, 6.66])/sum(np.array([0.79, 6.66]))*100, with_stddev = [0, 11]),
    'Pig_Iron_Consumption': dir_prior(np.array([0.94, 0.27, 0.24, 0.13])/sum(np.array([0.94, 0.27, 0.24, 0.13]))*100,  with_stddev = [0, 28]),

    'Basic_Oxygen_Furnace': dir_prior(np.array([22.20,  3.44])/sum(np.array([22.20,  3.44])) * 100, with_stddev = [0, 6]),
    'Electric_Arc_Furnace': dir_prior(np.array([22.20,  3.44])/sum(np.array([22.20,  3.44])) * 100, with_stddev = [0, 6]),
   
    'Continuous_Casting'  : dir_prior(np.array([30, 1.55])/sum(np.array([30, 1.55])) * 100, with_stddev = [0, 4]),
    'Ingot_Casting'       : dir_prior(np.array([30, 1.55])/sum(np.array([30, 1.55])) * 100, with_stddev = [0, 4]),
    'Other_Casting'       : dir_prior(np.array([30, 1.55])/sum(np.array([30, 1.55])) * 100, with_stddev = [0, 4]),
    
    'CC_Loss'             : dir_prior(np.array([30,  4.43])/sum(np.array([30,  4.43])) * 100, with_stddev = [0, 6]),
    'IC_Loss'             : dir_prior(np.array([30,  4.43])/sum(np.array([30,  4.43])) * 100, with_stddev = [0, 6]),
    'OC_Loss'             : dir_prior(np.array([30,  4.43])/sum(np.array([30,  4.43])) * 100, with_stddev = [0, 6]),
    'CC_Yield'            : dir_prior(np.array([11.46, 2.11, 2.82, 1.81])/sum(np.array([11.46, 2.11, 2.82, 1.81])) * 100, with_stddev = [0, 11]),

    'Plate_Mill'          : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Tin_Mill'            : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Hot_Strip_Mill'      : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Cold_Rolling_Mill'   : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Rod_and_Bar_Mill'    : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),

    'Section_Mill'        : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Primary_Mill'        : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Galvanized_Plant'    : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Pipe_Welding_Plant'  : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),
    'Seamless_Tube_Plant' : dir_prior(np.array([30,  2.73])/sum(np.array([30,  2.73]))* 100, with_stddev = [0, 5]),

}


observations = [
[          
(['Iron_Ore_Production'],       ['Export'],               11.2 ),  
(['Iron_Ore_Consumption'],      ['Blast_Furnace'],        46.3), 
(['Blast_Furnace'],             ['Pig_Iron'],             32.1), 
 
(['DRI'],                       ['Export'],               0.01),  
(['DRI_Consumption'],           ['Blast_Furnace'],        0.049), 
(['DRI_Consumption'],           ['Basic_Oxygen_Furnace'], 1.91), 
(['DRI_Consumption'],           ['Electric_Arc_Furnace'], 1.62), 
(['DRI_Consumption'],           ['Cupola'],               0.01 ), 
(['DRI_Consumption'],           ['Other_Casting'],        0.01 ),

(['Pig_Iron'],                  ['Export'],               0.021),  
(['Pig_Iron_Consumption'],      ['Basic_Oxygen_Furnace'], 31.5), 
(['Pig_Iron_Consumption'],      ['Electric_Arc_Furnace'], 5.79), 
(['Pig_Iron_Consumption'],      ['Cupola'],               0.057), 
(['Pig_Iron_Consumption'],      ['Other_Casting'],        0.046),

(['Scrap_Collected'],           ['Export'],               21.4 ), 
(['Scrap_Consumption'],         ['Blast_Furnace'],        2.62 ),          
(['Scrap_Consumption'],         ['Basic_Oxygen_Furnace'], 8.35 ) ,  
(['Scrap_Consumption'],         ['Electric_Arc_Furnace'], 50.9) ,      
(['Scrap_Consumption'],         ['Cupola'],               1.11 ),                    
(['Scrap_Consumption'],         ['Other_Casting'],        0.167),

(['Basic_Oxygen_Furnace'],      ['Continuous_Casting'],   36.281),        
(['Electric_Arc_Furnace'],      ['EAF_Yield'],            52.414),        
(['Pipe_Welding_Plant'],        ['Pipe_and_Tubing'],      2.165),                   
(['Seamless_Tube_Plant'],       ['Pipe_and_Tubing'],      2.162),

(['HSM_Yield'],                 ['Hot_Rolled_Sheet'],      19.544),    
(['CRM_Yield'],                 ['Cold_Rolled_Sheet'],     11.079),    
(['Tin_Mill'],                  ['Tin_Mill_Products'],     2.009),
(['Galvanized_Plant'],          ['Galvanized_Sheet'],      16.749),
(['Plate_Mill'],                ['Plates'],                 9.12),

(['RBM_Yield'],                 ['Reinforcing_Bars'],           5.65),
(['RBM_Yield'],                 ['Bars'],                       6.7),
(['RBM_Yield'],                 ['Wire_and_Wire_Rods'],         2.784),
(['RBM_Yield'],                 ['Light_Section'],              2.13 ),   
(['SM_Yield'],                  ['Heavy_Section'],              5.03),   
(['SM_Yield'],                  ['Rails_and_Rail_Accessories'], 1.009), 
],
]

observations_ratio = [
#12
[     
(['Cold_Rolled_Sheet'], ['Automotive'],        0.250),  
(['Cold_Rolled_Sheet'], ['Machinery'],         0.079),  
(['Cold_Rolled_Sheet'], ['Steel_Products'],    0.313), 
(['Cold_Rolled_Sheet'], ['Export'],            0.112), 
 
(['Galvanized_Sheet'],  ['Construction'],      0.19),  
(['Galvanized_Sheet'],  ['Automotive'],        0.42), 
(['Galvanized_Sheet'],  ['Export'],            0.15),

(['Hot_Rolled_Sheet'],  ['Construction'],      0.59 ),  
(['Hot_Rolled_Sheet'],  ['Automotive'],        0.133 ), 
(['Hot_Rolled_Sheet'],  ['Machinery'],         0.108 ),  
(['Hot_Rolled_Sheet'],  ['Energy'],            0.01  ), 
(['Hot_Rolled_Sheet'],  ['Steel_Products'],    0.0027), 
(['Hot_Rolled_Sheet'],  ['Export'],            0.065),  
    
(['Pipe_and_Tubing'],   ['Construction'],       0.227), 
(['Pipe_and_Tubing'],   ['Automotive'],         0.08), 
(['Pipe_and_Tubing'],   ['Machinery'],          0.04),  
(['Pipe_and_Tubing'],   ['Energy'],             0.55), 
(['Pipe_and_Tubing'],   ['Export'],             0.065),


(['Plates'],            ['Construction'],       0.0408), 
(['Plates'],            ['Automotive'],         0.01), 
(['Plates'],            ['Machinery'],          0.5187),  
(['Plates'],            ['Energy'],             0.067), 
(['Plates'],            ['Export'],             0.231),  
     
(['Bars'],              ['Construction'],       0.152), 
(['Bars'],              ['Automotive'],         0.311), 
(['Bars'],              ['Machinery'],          0.238),  
(['Bars'],              ['Energy'],             0.046), 
(['Bars'],              ['Export'],             0.131),    
    
(['Reinforcing_Bars'],  ['Construction'],       0.925),
(['Reinforcing_Bars'],  ['Export'],             0.039),

(['Tin_Mill_Products'], ['Automotive'],         0.006),
(['Tin_Mill_Products'], ['Steel_Products'],     0.685),
(['Tin_Mill_Products'], ['Export'],             0.067),
     
(['Wire_and_Wire_Rods'],  ['Construction'],     0.388), 
(['Wire_and_Wire_Rods'],  ['Automotive'],       0.285), 
(['Wire_and_Wire_Rods'],  ['Machinery'],        0.1),  
(['Wire_and_Wire_Rods'],  ['Energy'],           0.049), 
(['Wire_and_Wire_Rods'],  ['Export'],           0.094),     
     
(['Rails_and_Rail_Accessories'],  ['Construction'],  0.779), 
(['Rails_and_Rail_Accessories'],  ['Machinery'],     0.047), 
(['Rails_and_Rail_Accessories'],  ['Export'],        0.141),

(['Light_Section'],          ['Construction'],   0.86),
(['Light_Section'],          ['Automotive'],     0.026),
(['Light_Section'],          ['Export'],         0.057),
    
(['Heavy_Section'],          ['Construction'],   0.877),
(['Heavy_Section'],          ['Export'],         0.092),

(['Steel_Product_Casting'],  ['Construction'],   0.259), 
(['Steel_Product_Casting'],  ['Automotive'],     0.385), 
(['Steel_Product_Casting'],  ['Machinery'],      0.259), 

(['Iron_Product_Casting'],   ['Construction'],   0.311), 
(['Iron_Product_Casting'],   ['Automotive'],     0.552), 
(['Iron_Product_Casting'],   ['Machinery'],      0.066),  
],
    
]


sorted(list(input_defs[0].keys()))

class SplitParamModel:
    """Flow model with different types of prior for different process sub-models.
    """
    def __init__(self, processes, input_defs, param_defs, input_mu, input_sigma, input_lower, input_upper, flow_observations=None, ratio_observations = None,
                 input_observations=None, inflow_observations=None):
        self.processes = processes
        self.possible_inputs = possible_inputs = sorted(list(input_defs[0].keys()))
        self.param_defs = param_defs

    with pm.Model() as self.model:
            if flow_observations is not None:
                sigma       = pm.TruncatedNormal('sigma',       mu = 0, sigma = 0.15, lower = 0, upper = 0.5, shape = len(flow_observations[0]))
            
            if input_observations is not None:
                sigma_input = pm.TruncatedNormal('sigma_input', mu = 0, sigma = 0.15, lower = 0, upper = 0.5, shape = len(input_observations[0]))

            if ratio_observations is not None:                
                sigma_ratio = pm.TruncatedNormal('sigma_ratio', mu = 0, sigma = 0.15, lower = 0, upper = 0.5, shape = len(ratio_observations[0]))

            process_params = {
                    pid: process.param_rv(pid, param_defs.get(pid))
                    for pid, process in processes.items() 
                }
            for i in range(len(input_defs)):
                inputs = pm.TruncatedNormal('inputs_{}'.format(i), mu = input_mu, sigma = input_sigma, lower = input_lower,
                                           upper = input_upper, shape = 8)
                
                transfer_coeffs, all_inputs = self._build_matrices(process_params, inputs)
                transfer_coeffs = pm.Deterministic('TCs_coeffs_{}'.format(i), transfer_coeffs)
                process_throughputs = pm.Deterministic(
                    'X_{}'.format(i), T.dot(matrix_inverse(T.eye(len(processes)) - transfer_coeffs), all_inputs))

                flows = pm.Deterministic('F_{}'.format(i), transfer_coeffs.T * process_throughputs[:, None])

                if flow_observations is not None:
                    flow_obs, flow_data = self._flow_observations(flow_observations[i])
                    Fobs = pm.Deterministic('Fobs_{}'.format(i), T.tensordot(flow_obs, flows, 2))
                    pm.Normal('FD_{}'.format(i), mu= Fobs, sd=Fobs*sigma, observed = flow_data )

                if input_observations is not None:
                    input_obs, input_data = self._input_observations(input_observations[i])
                    Iobs = pm.Deterministic('Iobs_{}'.format(i), T.dot(input_obs, all_inputs))
                    pm.Normal('ID_{}'.format(i), mu=Iobs, sd= Iobs*sigma_input, observed=input_data)

                if ratio_observations is not None:
                    ratio_obs, ratio_data = self._flow_observations(ratio_observations[i])
                    Robs     = pm.Deterministic('Robs_{}'.format(i), T.tensordot(ratio_obs,  transfer_coeffs.T, 2))
                    pm.Normal('RD_{}'.format(i), mu= Robs, sd= Robs*sigma_ratio, observed = ratio_data )

    def _build_matrices(self, process_params, inputs):
        Np = len(self.processes)
        transfer_coeffs = T.zeros((Np, Np))

        pids = {k: i for i, k in enumerate(self.processes)}
        for pid, process in self.processes.items():
            if not process_params.get(pid):
                continue
            params = process_params[pid]
            process_tcs = process.transfer_functions(params)
            if process.outputs:
                dest_idx = [pids[dest_id] for dest_id in process.outputs]
                transfer_coeffs = T.set_subtensor(transfer_coeffs[dest_idx, pids[pid]], process_tcs)

        possible_inputs_idx = [pids[k] for k in self.possible_inputs]
        all_inputs = T.zeros(Np)
        all_inputs = T.set_subtensor(all_inputs[possible_inputs_idx], inputs)

        return transfer_coeffs, all_inputs


    def _flow_observations(self, observations):
        Np = len(self.processes)
        No = len(observations)
        flow_obs = np.zeros((No, Np, Np))
        flow_data = np.zeros(No)
        pids = {k: i for i, k in enumerate(self.processes)}
        for i, (sources, targets, value) in enumerate(observations):
            flow_obs[i, [pids[k] for k in sources], [pids[k] for k in targets]] = 1
            flow_data[i] = value
        return flow_obs, flow_data

    def _input_observations(self, observations):
        Np = len(self.processes)
        No = len(observations)
        input_obs = np.zeros((No, Np))
        input_data = np.zeros(No)
        pids = {k: i for i, k in enumerate(self.processes)}
        for i, (targets, value) in enumerate(observations):
            input_obs[i, [pids[k] for k in targets]] = 1
            input_data[i] = value

        return input_obs, input_data


input_mu    = np.array([3.18,  4.15,   3.69, 3.41,  9.1,    25,   47.87,  71.74])    #Elicited from one expert only
input_sigma = np.array([1.48,  1.27 ,  1.7,  1.75,  3.94,   10,   9.69,   7.49])
input_lower = np.array([0,     2,      2.5,  2,     0,      10,   20,     50 ])
input_upper = np.array([5,     7,      7,    6,     20,     45,   70,     90])


Model     = SplitParamModel(processes, input_defs, param_defs0,  input_mu, input_sigma, input_lower, input_upper, flow_observations= observations, input_observations= input_observations, ratio_observations = observations_ratio)


# with Model.model:
#     Trace = pm.sample_smc(draws = 10000, n_steps = 150, chains = 2, cores = 2, tune_steps=True, p_acc_rate = 0.99, random_seed = 12)
#     ps = pm.summary(Trace)

