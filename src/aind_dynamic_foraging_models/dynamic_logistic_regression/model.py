import psytrack as psy

from aind_analysis_arch_result_access.han_pipeline import get_logistic_regression

'''
1. get NWB file
2. get static weight
3. Build design matrix
4. fit
'''

def make_design_matrix():
    return

def fit():
    return

def get_static_weights(session_date, subject):
    df_logistic = get_logistic_regression(
        df_sessions = pd.DataFrame(
            {
                "subject_id":[subject],
                "session_date":[session_date],
            }
        ),
        model='Su2022'
        )
    return df_logistic
