import logging

import numpy as np
import torch

seed = 11
np.random.seed(seed)
torch.manual_seed(seed)

from preprocessing import num_objects, num_events, num_unique_acts, remaining_time, ocel_to_csv
from major_function import main_function

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # Create list of filenames and corresponding object types to use
    names_ot_dict = {"recruiting-ocel1": ['applicants', 'applications', 'offers'],
                     "BPI2017-Final": ["application", "offer"],
                     "socel2_hinge": ['SteelSheet', 'FormedPart', 'FemalePart', 'MalePart', 'SteelPin', 'Hinge',
                                      'HingePack'],
                     "ocel2-p2p": ['material', 'purchase_requisition', 'quotation', 'purchase_order', 'goods_receipt',
                                   'invoice_receipt', 'payment']
                     }

    # This is the order management event log
    # filename = "orders.jsonocel" filename = "ContainerLogistics"
    # #parameters = {"execution_extraction": "leading_type",
    # #             "leading_type": "items"}
    # ocel = ocel_import_factory.apply(filename)#, parameters=parameters)

    # Target extraction functions
    subg_funcs = []
    g_funcs = [num_events]
    # Set this value to:
    # - True to use the graph structure only as input
    # - False to use event-level features detailed in main_function
    no_feats = [False]
    ks = range(11, 12)
    embedding_name_list = ["GAT"] #, "GCN", "Graph2Vec", "FGSD"] # "GAT"
    embedding_size_list = [8, 16, 64]
    batch_size = 32
    learning_rate = 0.01
    mlp_hidden_dim_list = [8]#, 32]
    mlp_num_layers_list = [1]#, 4]
    filename = "recruiting-ocel1"
    object_types = names_ot_dict[filename]
    parameters = {"obj_names": object_types,
                  "val_names": [],
                  "act_name": "activity",
                  "time_name": "timestamp",
                  "sep": ","}
    for embedding_name in embedding_name_list:
        for embedding_size in embedding_size_list:
            for mlp_hidden_dim in mlp_hidden_dim_list:
                for mlp_num_layers in mlp_num_layers_list:
                    for no_feat in no_feats:
                        main_function(filename, parameters, subg_funcs, g_funcs, no_feat, ks, embedding_name,
                                      embedding_size, batch_size, learning_rate, mlp_hidden_dim, mlp_num_layers)
    a = 0
