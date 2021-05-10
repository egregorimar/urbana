import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    mutual_info_regression,
    f_regression,
    SelectPercentile,
)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    RepeatedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


from urbana.features import normaltests
from urbana.constants import DIR_REPO, DIR_DATA, RANDOM_STATE
from urbana.models.plot_predictions import PredictedAccuracy


def LinearModel(YEAR, MONTH, VARIABLE_TO_PREDICT):

    mpl.use('Agg')

    print("Starting with " + str(YEAR) + "-" + str(MONTH))


    OUTPUT_WARNINGS = False
    SAVE_FIGS = True
    SAVE_MODEL = True

    K_EDUCATION = 1
    K_AGE = 2
    K_NATIONALITY = 2
    K_RENT = 1
    K_POI = 10


    if not OUTPUT_WARNINGS:
        import warnings

        warnings.filterwarnings("ignore")


    # Create folders to store the data

    DIR_VAR = DIR_DATA / "processed/{}".format(VARIABLE_TO_PREDICT)
    DIR_MONTH = DIR_VAR / "{}_{:02d}".format(YEAR, MONTH)
    DIR_LINEAR = DIR_MONTH / "01_linear"

    if SAVE_FIGS or SAVE_MODEL:
        folder_list = [DIR_VAR, DIR_MONTH, DIR_LINEAR, DIR_VAR / "01_linear"]

        import os

        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)


    np.random.seed(RANDOM_STATE)

    sect = pd.read_csv(
        DIR_DATA / "interim/sections_{}_{:02d}.csv".format(int(str(YEAR)[2:4]), MONTH),
    )

    sect.set_index("Tag", inplace=True)

    sect.drop(["N_district", "N_neighbourhood", "N_section"], axis=1, inplace=True)



    y = sect[VARIABLE_TO_PREDICT]

    X = sect.drop(
        ["Airbnb_Number", "Airbnb_Price", "Airbnb_Price_Person", "Airbnb_Location_Score"],
        axis=1,
    )



    geo_info = gpd.read_file(DIR_DATA / "interim/sections_geo.json")

    geo_info.set_index("Tag", inplace=True)

    geo_info[VARIABLE_TO_PREDICT] = sect[VARIABLE_TO_PREDICT]

    #print("Area with maximum value: " + str(geo_info[VARIABLE_TO_PREDICT].idxmax()))



    fig, ax = plt.subplots(figsize=(20, 20))

    geo_info.plot(
        ax=ax,
        column=VARIABLE_TO_PREDICT,
        legend=True,
        figsize=(20, 20),
        legend_kwds={"shrink": 0.7},
    )

    ax.set_title("Target variable: " + str(VARIABLE_TO_PREDICT), fontsize=20, y=1.01)

    if SAVE_FIGS:
        plt.savefig(DIR_LINEAR / "target_variable.svg", format="svg")


    # # # First model: All features

    # # ## Pipeline fit
    # pipe_all = Pipeline(
    #     steps=[
    #         ("imputer", IterativeImputer()),
    #         ("regressor", LinearRegression()),
    #     ]
    # )
    # pipe_all.fit(X_train, y_train)

    # y_test_pred_all = pipe_all.predict(X_test)
    # # ## Prediction plot
    # pa_all = PredictedAccuracy(y_test, y_test_pred_all)
    # pa_all.plot_scatter()
    # # ## Sensitivity analysis
    # cv_all = cross_validate(
    #     pipe_all,
    #     X,
    #     y,
    #     cv=RepeatedKFold(n_splits=5, n_repeats=5),
    #     scoring=["neg_root_mean_squared_error"],
    #     return_estimator=True,
    #     n_jobs=-1,
    # )

    # coefs_all = pd.DataFrame(
    #     [est.named_steps["regressor"].coef_ for est in cv_all["estimator"]],
    #     columns=X.columns,
    # )

    # medians_all = coefs_all.median()
    # medians_all = medians_all.reindex(medians_all.abs().sort_values(ascending=False).index)
    # coefs_all = coefs_all[medians_all.index]


    # plt.figure(figsize=(20, 40))
    # sns.stripplot(data=coefs_all, orient="h", color="k", alpha=0.5)
    # sns.boxplot(data=coefs_all, orient="h", color="cyan", saturation=0.5)
    # plt.axvline(x=0, color=".5")
    # # Second model: Transformations and feature selection

    # ## Normality tests


    # Check which variables are already normal
    normality_test = normaltests.get_normaltest_df(X.T)

    #print(normality_test["dagostino"].value_counts())
    #print(normality_test["shapiro"].value_counts())


    # ## Preprocessing Pipeline

    # The preprocessing will have two phases:
    # * KNNImputer: To imput missing data by fitting knn (better than imputing the mean or the median)
    # * PowerTransformer: Since none of the features is Gaussian, we will transform them with the *Yeo-Johnson* transformation


    pt = PowerTransformer()
    preprocessor = Pipeline(steps=[("imputer", KNNImputer()), ("pt", pt)])


    # ## Feature Selection by Subgroups



    X_Education = X.filter(regex="^Education")

    kbest_Education = SelectKBest(f_regression, k=K_EDUCATION).fit(
        preprocessor.fit_transform(X_Education),
        pt.fit_transform(y.values.reshape(-1, 1)),
    )

    education_cols = kbest_Education.get_support(indices=True)
    X_Education_chosen = X_Education.columns[education_cols]
    #X_Education_chosen



    X_Age = X.filter(regex="^Percentage_Age_")
    kbest_Age = SelectKBest(f_regression, k=K_AGE).fit(
        preprocessor.fit_transform(X_Age),
        pt.fit_transform(y.values.reshape(-1, 1)),
    )

    age_cols = kbest_Age.get_support(indices=True)
    X_Age_chosen = X_Age.columns[age_cols]
    #X_Age_chosen



    X_Nationality = X.filter(regex="^Nationality_")
    X_Nationality.drop(["Nationality_Spain"], axis=1, inplace=True)
    kbest_Nationality = SelectKBest(f_regression, k=K_NATIONALITY).fit(
        preprocessor.fit_transform(X_Nationality),
        pt.fit_transform(y.values.reshape(-1, 1)),
    )

    nationality_cols = kbest_Nationality.get_support(indices=True)
    X_Nationality_chosen = X_Nationality.columns[nationality_cols]
    #X_Nationality_chosen

    if YEAR >= 2015 and YEAR <= 2018:

        X_Rent = X.filter(regex="^Rent_")
        kbest_Rent = SelectKBest(f_regression, k=K_RENT).fit(
            preprocessor.fit_transform(X_Rent),
            pt.fit_transform(y.values.reshape(-1, 1)),
        )

        rent_cols = kbest_Rent.get_support(indices=True)
        X_Rent_chosen = X_Rent.columns[rent_cols]
        #X_Rent_chosen




    X_POI = X.filter(regex="^POI")

    kbest_POI = SelectKBest(f_regression, k=K_POI).fit(
        preprocessor.fit_transform(X_POI),
        pt.fit_transform(y.values.reshape(-1, 1)),
    )

    POI_cols = kbest_POI.get_support(indices=True)
    X_POI_chosen = X_POI.columns[POI_cols]
    #X_POI_chosen


    if YEAR >= 2015 and YEAR <= 2018:
        X.drop(np.setdiff1d(X_Rent.columns, X_Rent_chosen), axis=1, inplace=True)

    X.drop(np.setdiff1d(X_Age.columns, X_Age_chosen), axis=1, inplace=True)

    X.drop(np.setdiff1d(X_Nationality.columns, X_Nationality_chosen), axis=1, inplace=True)

    X.drop(np.setdiff1d(X_Education.columns, X_Education_chosen), axis=1, inplace=True)

    X.drop(np.setdiff1d(X_POI.columns, X_POI_chosen), axis=1, inplace=True)


    # ## Feature Selection Pipeline

    # In order to perform a feature selection, we will use *RFE* (Recursive Feature Elimination).
    # 
    # The number of variables to use will be a hyper-paramater that will be tuned with a GridSearch using RMSE as the metric.
    # 
    # The target feature will also be transformed with a PowerTransfomrer, by applying the *TransformedTargetRegressor*.


    # Define the regressor to use
    myRegressor = LinearRegression()

    # Define a pipeline with the preprocessing, feature selection (RFE) and regressor
    pipe_rfe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rfe", RFE(estimator=myRegressor)),
            ("regressor", myRegressor),
        ]
    )

    # Define the param space for hyper-parameter tunning (in this case, the number of features to keep with RFE)
    param_grid_rfe = [{"rfe__n_features_to_select": np.arange(6, 15, 1)}]

    search_rfe = GridSearchCV(
        pipe_rfe, param_grid_rfe, scoring="neg_root_mean_squared_error", n_jobs=-1
    )


    model = TransformedTargetRegressor(regressor=search_rfe, transformer=PowerTransformer())

    model.fit(X, y)



    #print("Best Model:")
    #print(
    #    "Number of features: "
    #    + str(model.regressor_.best_params_["rfe__n_features_to_select"])
    #)
    #print("\nList of features:")
    cols_rfe = model.regressor_.best_estimator_.named_steps["rfe"].get_support(indices=True)
    #print(X.columns[cols_rfe])



    score_features = -model.regressor_.cv_results_["mean_test_score"]
    n_features = []
    for i in model.regressor_.cv_results_["params"]:
        n_features.append(i["rfe__n_features_to_select"])

    id_min_score = score_features.argmin()

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.plot(n_features, score_features, marker="o")
    plt.axvline(x=n_features[id_min_score], color=".5")

    ax.set_xlabel("Number of features", fontsize=15)
    ax.set_ylabel("Median Absolute Error", fontsize=15)
    ax.set_xticks(np.arange(min(n_features), max(n_features) + 1))
    ax.set_title("Score by number of features", fontsize=20, y=1.01)

    if SAVE_FIGS:
        plt.savefig(DIR_LINEAR / "selection_rmse.svg", format="svg")

    plt.close()



    y_pred_rfe = model.predict(X).round()
    pa_rfe = PredictedAccuracy(y, y_pred_rfe)
    pa_rfe.plot_scatter(save_fig=SAVE_FIGS, root_name=DIR_LINEAR / "model")
    pa_rfe.plot_errors(save_fig=SAVE_FIGS, root_name=DIR_LINEAR / "model")


    del pa_rfe

    geo_info["Chosen_Error"] = 2 * (y - y_pred_rfe) / (abs(y) + abs(y_pred_rfe))

    col_lim = max(abs(geo_info["Chosen_Error"].min()), abs(geo_info["Chosen_Error"].max()))

    fig, ax = plt.subplots(figsize=(20, 20))

    geo_info.plot(
        ax=ax,
        column="Chosen_Error",
        legend=True,
        figsize=(20, 20),
        edgecolor="black",
        cmap="coolwarm",
        vmin=-col_lim,
        vmax=col_lim,
        legend_kwds={"shrink": 0.7},
    )

    ax.set_title("Relative errors in linear model", fontsize=20, y=1.01)


    if SAVE_FIGS:
        plt.savefig(DIR_LINEAR / "relative_errors.svg", format="svg")

    plt.close()

    if SAVE_MODEL:
        geo_info[["Chosen_Error"]].to_csv(DIR_LINEAR / "relative_errors.csv")
        df_predictions = pd.DataFrame(y_pred_rfe, index=geo_info.index, columns=["Predictions"])
        df_predictions.to_csv(DIR_LINEAR / "predictions.csv")  
        df_predictions.to_csv(DIR_VAR / "01_linear/{}_{:02d}_predictions.csv".format(YEAR, MONTH))  

    # from yellowbrick.regressor.residuals import residuals_plot

    # residuals_plot(
    #     model, X_train, y_train, X_test, y_test, hist=False, qqplot=True, is_fitted=True
    # )

    # residuals_plot(
    #     model, X_train, y_train, X_test, y_test, hist=True, qqplot=False, is_fitted=True
    # )


    pw = PowerTransformer()
    pw.fit(y.values.reshape(-1, 1))

    ####################Tranform y_hat####################
    y_pred_transformed = model.predict(X)
    y_pred_transformed = pw.transform(y_pred_transformed.reshape(-1, 1)).flatten()

    ####################Trasform y####################
    # y_test_transformed = pd.Series(pw.transform(y_test.values.reshape(-1, 1)).flatten())
    y_transformed = pd.Series(pw.transform(y.values.reshape(-1, 1)).flatten())
    y_transformed.name = "Transformed Airbnb_Number"

    pa_trans = PredictedAccuracy(y_transformed, y_pred_transformed)
    pa_trans.plot_scatter(
        save_fig=SAVE_FIGS,
        root_name=DIR_LINEAR / "transformed_model",
    )
    pa_trans.plot_errors(
        save_fig=SAVE_FIGS,
        root_name=DIR_LINEAR / "transformed_model",
    )
    del pa_trans


    # # Sensitivity Analysis


    X_rfe = X.iloc[:, cols_rfe]

    pipe_sens = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", myRegressor)])

    model_sens = TransformedTargetRegressor(
        regressor=pipe_sens, transformer=PowerTransformer()
    )

    model_sens.fit(X_rfe, y)



    cv_rfe = cross_validate(
        model_sens,
        X_rfe,
        y,
        cv=RepeatedKFold(n_splits=5, n_repeats=5),
        scoring=["neg_root_mean_squared_error"],
        return_estimator=True,
        n_jobs=-1,
    )

    coefs_rfe = pd.DataFrame(
        [est.regressor_.named_steps["regressor"].coef_ for est in cv_rfe["estimator"]],
        columns=X_rfe.columns,
    )


    coefs_rfe["Intercept"] = pd.Series(
        [est.regressor_.named_steps["regressor"].intercept_ for est in cv_rfe["estimator"]]
    )

    medians_rfe = coefs_rfe.drop(["Intercept"], axis=1).median()
    medians_rfe = medians_rfe.reindex(medians_rfe.abs().sort_values(ascending=False).index)
    medians_rfe = medians_rfe.append(pd.Series({"Intercept": 0}, index=["Intercept"]))
    coefs_rfe = coefs_rfe[medians_rfe.index]

    limit_value = (
        max(abs(coefs_rfe.to_numpy().min()), abs(coefs_rfe.to_numpy().max())) * 1.05
    )



    fig, ax = plt.subplots(figsize=(20, 20))

    sns.stripplot(ax=ax, data=coefs_rfe, orient="h", color="k", alpha=0.5)
    sns.boxplot(ax=ax, data=coefs_rfe, orient="h", color="cyan", saturation=0.5)
    plt.axvline(x=0, color="red")

    plt.figtext(0.51, 0.9, "Linear Model: Coefficient robustness", fontsize=20, ha="center")
    plt.figtext(
        0.51,
        0.885,
        "{}-{:02d}".format(YEAR, MONTH),
        fontsize=18,
        ha="center",
    )
    ax.set_xlim(-limit_value, limit_value)
    ax.set_xlabel("Coefficient value", fontsize=15)

    if SAVE_FIGS:
        plt.savefig(DIR_LINEAR / "sensitivity.svg", format="svg")
        plt.savefig(DIR_VAR / "01_linear/{}_{:02d}_sensitivity.svg".format(YEAR, MONTH), format="svg")
        

    plt.close()




    if SAVE_MODEL:
        coefs_rfe.to_csv(DIR_LINEAR / "coefficients.csv")
        coefs_rfe.to_csv(DIR_VAR / "01_linear/{}_{:02d}_coefficients.csv".format(YEAR, MONTH))

    print("Done with " + str(YEAR) + "-" + str(MONTH))
    print("##################################")
