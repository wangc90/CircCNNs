from inference import seq_to_tensor, BS_LS_upper_lower, BS_LS_upper_lower_concat, BS_LS_pred
from pre_trained_model_structure import *
### using the function defined in inference.py to do prediction to new exon pairs


def predictions(testing_df_path, trained_model):
    '''
    :return: predicted circRNA (1) or linear RNA(0) with associated probabilities
    '''
    if trained_model == 1:
        print('Predicting the circRNA with trained base model1')
        ### prepare the data for base model1
        seq_upper_lower_concate_feature = seq_to_tensor(testing_df_path=testing_df_path,
                                                        is_upper_lower_concat=True)
        test_data_model1 = BS_LS_upper_lower_concat(seq_upper_lower_concate_feature)
        ### brining in trained model weight for base model1
        base_model1_10000_path = "/home/wangc90/circRNA/CircCNNs/Trained_Model_Weights/Base_model1_retraining_10000/retrained_model_149.pt"

        ### do prediction based on base_model1
        base1_pred_labels, base1_pred_probs = BS_LS_pred(dataset=test_data_model1,
                                                         model_path=base_model1_10000_path,
                                                         model_type=1)
        print("These are the predicted labels")
        print(base1_pred_labels)
        print("These are the predicted probabilities for the corresponding label being circRNA")
        print(base1_pred_probs)

    elif trained_model == 2:
        print('Predicting the circRNA with trained base model2')
        ### prepare the data for base model2
        seq_upper_feature, seq_lower_feature = seq_to_tensor(testing_df_path=testing_df_path,
                                                             is_upper_lower_concat=False)
        test_data_model2 = BS_LS_upper_lower(seq_upper_feature, seq_lower_feature)
        ### brining in trained model weight for base model2
        base_model2_10000_path = "/home/wangc90/circRNA/CircCNNs/Trained_Model_Weights/Base_model2_retraining_10000/retrained_model_149.pt"

        ### do prediction based on base_model2
        base2_pred_labels, base2_pred_probs = BS_LS_pred(dataset=test_data_model2,
                                                         model_path=base_model2_10000_path,
                                                         model_type=2)

        print("These are the predicted labels")
        print(base2_pred_labels)
        print("These are the predicted probabilities for the corresponding label being circRNA")
        print(base2_pred_probs)


if __name__ == "__main__":
    testing_df_path='/home/wangc90/circRNA/CircCNNs/Data/testing_set_keys.csv'
    predictions(testing_df_path=testing_df_path,trained_model=1)
    predictions(testing_df_path=testing_df_path,trained_model=2)
