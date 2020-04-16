from matplotlib import pyplot as plt
from kneed import DataGenerator, KneeLocator

from utils.metrics_util import evaluate
import pandas as pd


def plot(summary_df, r0):
    print(summary_df)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(summary_df['date'], summary_df['predicted_count'], color='#701805', label="infections_r0_{}".format(r0))
    ax.plot(summary_df['date'], summary_df['hospitalizations_count'], color='#cf563e',
            label="hospitalizations_r0_{}".format(r0))
    ax.plot(summary_df['date'], summary_df['ICU_count'], color='#e39191', label="ICU requirement_r0_{}".format(r0))
    ax.plot(summary_df['date'], summary_df['ventilator_count'], color='#cf823e',
            label="ventilators requirement_r0_{}".format(r0))
    ax.plot(summary_df['date'], summary_df['fatality_count'], color='#bdb56d', label="fatalities_r0_{}".format(r0))
    ax.plot(summary_df['date'], summary_df['actual_count'], color='#0000FF', label="Actual Count")
    plt.xlabel("Time")
    plt.ylabel("SEIR Projections")
    plt.title("region")
    ax.legend()


# Needs predict dataframe to
# have 3 columns, date, actual_count, predicted_count

def generate_summary_data(predict_df, F_hospitalized, F_need_ICU, F_need_Ventilator, F_fatality):
    summary_df = predict_df
    summary_df['hospitalizations_count'] = (F_hospitalized * predict_df['predicted_count']).round()
    summary_df['ICU_count'] = (F_need_ICU * predict_df['predicted_count']).round()
    summary_df['ventilator_count'] = (F_need_Ventilator * predict_df['predicted_count']).round()
    summary_df['fatality_count'] = (F_fatality * predict_df['predicted_count']).round()
    summary_df['fatality_count'] = (F_fatality * predict_df['predicted_count']).round()
    summary_df['peak_date'] = summary_df.iloc[summary_df['predicted_count'].idxmax()]['date']
    summary_df['peak_infections'] = summary_df.iloc[summary_df['predicted_count'].idxmax()]['predicted_count']
    metrics = evaluate(predict_df["actual_count"], predict_df["predicted_count"])
    for key in list(metrics.keys()):
        summary_df[key] = metrics[key]
    kneedle = KneeLocator(summary_df.index, summary_df['predicted_count'], S=1.0, curve='convex',
                          direction='increasing')
    if (pd.notnull(kneedle.knee)):
        summary_df['knee_point_date'] = summary_df.iloc[kneedle.knee, summary_df.columns.get_loc('date')]
    else:
        summary_df['knee_point_date'] = None

    return summary_df
