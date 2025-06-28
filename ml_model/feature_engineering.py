import pandas as pd

def create_features(df):
    df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['transaction_hour'] = df['step'] % 24
    df['transaction_day'] = df['step'] // 24
    df['amount_ratio_sender'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['sender_zero_balance'] = (df['newbalanceOrig'] == 0).astype(int)
    df['receiver_zero_before'] = (df['oldbalanceDest'] == 0).astype(int)
    df['is_exact_transfer'] = (df['balance_change_dest'] == df['amount']).astype(int)
    df['type_encoded'] = df['type'].astype('category').cat.codes

    df['isPotentialFraud'] = (
        (df['amount_ratio_sender'] < 1.05) &
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
        ((df['newbalanceOrig'] == 0) | (df['oldbalanceDest'] == 0) | (df['newbalanceDest'] == 0))
    ).astype(int)

    suspicious_destinations = df.loc[df['isFraud'] == 1, 'nameDest'].unique()
    df['isSuspiciousDest'] = df['nameDest'].isin(suspicious_destinations).astype(int)
    df['isSharedRiskyDest'] = (df['nameDest'].map(df['nameDest'].value_counts()) > 5).astype(int)

    df['riskNote'] = 'Clean'
    df.loc[df['isFraud'] == 1, 'riskNote'] = 'Fraudulent'
    df.loc[(df['isPotentialFraud'] == 1) & (df['isFraud'] == 0), 'riskNote'] = 'High-risk destination'
    df.loc[df['isSharedRiskyDest'] == 1, 'riskNote'] = 'Shared-risk destination'

    df['riskLevel'] = df['riskNote'].map({
        'Fraudulent': 2,
        'High-risk destination': 1,
        'Shared-risk destination': 1,
        'Clean': 0
    })

    return df
