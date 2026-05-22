#include "decision_tree.h"
#include "dataset.h"
#include "tree_visualization.h"

#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Параметры по умолчанию для классификации команд робота. */
static TreeParams default_classification_params(SplitCriterion criterion) {
    TreeParams params;
    params.max_depth = 7;
    params.min_samples_split = 4;
    params.max_features = 2;
    params.criterion = criterion;
    return params;
}

/* Параметры по умолчанию для регрессии безопасной скорости. */
static TreeParams default_regression_params(void) {
    TreeParams params;
    params.max_depth = 7;
    params.min_samples_split = 4;
    params.max_features = 2;
    params.criterion = CRITERION_MSE;
    return params;
}

/* Название критерия для вывода в консоль. */
static const char *criterion_name(SplitCriterion criterion) {
    if (criterion == CRITERION_INFO_GAIN) return "Information Gain / Entropy";
    if (criterion == CRITERION_GINI) return "Gini";
    return "MSE / Variance Reduction";
}

/* Очистка остатка строки после ввода пользователя. */
static void clear_input_line(void) {
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF) {
        /* пропускаем лишние символы */
    }
}

/* Безопасное чтение числа double. */
static double read_double_value(const char *prompt) {
    double value;
    while (1) {
        printf("%s", prompt);
        if (scanf("%lf", &value) == 1) {
            clear_input_line();
            return value;
        }
        printf("Input error. Please enter a number, for example 12.5\n");
        clear_input_line();
    }
}

/* Безопасное чтение целого числа. */
static int read_int_value(const char *prompt) {
    int value;
    while (1) {
        printf("%s", prompt);
        if (scanf("%d", &value) == 1) {
            clear_input_line();
            return value;
        }
        printf("Input error. Please enter an integer.\n");
        clear_input_line();
    }
}

/* Оценка точности классификации на обучающем датасете. */
static double evaluate_classification_accuracy(const RandomForest *forest, const Dataset *data) {
    int correct = 0;
    for (int i = 0; i < data->n_samples; i++) {
        int predicted = predict_forest_class(forest, data->X[i]);
        if (predicted == data->y_class[i]) correct++;
    }
    return 100.0 * (double)correct / (double)data->n_samples;
}

/* Оценка среднеквадратичной ошибки для регрессии. */
static double evaluate_regression_mse(const RandomForest *forest, const Dataset *data) {
    double sum = 0.0;
    for (int i = 0; i < data->n_samples; i++) {
        double predicted = predict_forest_regression(forest, data->X[i]);
        double diff = predicted - data->y_reg[i];
        sum += diff * diff;
    }
    return sum / (double)data->n_samples;
}

/* Обучение ансамбля деревьев для задачи классификации. */
static void train_classification(RandomForest *forest, const Dataset *data, SplitCriterion criterion) {
    if (forest->trees) {
        free_random_forest(forest);
    }

    TreeParams params = default_classification_params(criterion);
    int n_trees = 15;
    *forest = train_random_forest(data, TASK_CLASSIFICATION, params, n_trees);

    double accuracy = evaluate_classification_accuracy(forest, data);
    printf("\nClassification model has been trained.\n");
    printf("Split criterion: %s\n", criterion_name(criterion));
    printf("Number of bagging trees: %d\n", n_trees);
    printf("Random features per node: %d of %d\n", params.max_features, data->n_features);
    printf("Training accuracy: %.2f%%\n", accuracy);
}

/* Обучение ансамбля деревьев для задачи регрессии. */
static void train_regression(RandomForest *forest, const Dataset *data) {
    if (forest->trees) {
        free_random_forest(forest);
    }

    TreeParams params = default_regression_params();
    int n_trees = 15;
    *forest = train_random_forest(data, TASK_REGRESSION, params, n_trees);

    double mse = evaluate_regression_mse(forest, data);
    printf("\nRegression model has been trained.\n");
    printf("Split criterion: %s\n", criterion_name(params.criterion));
    printf("Number of bagging trees: %d\n", n_trees);
    printf("Random features per node: %d of %d\n", params.max_features, data->n_features);
    printf("Training MSE: %.5f\n", mse);
}


/* Экспорт первого дерева из ансамбля в DOT и PNG. */
static void export_first_tree_visualization(const RandomForest *forest,
                                            const Dataset *data,
                                            const char *model_name,
                                            const char *dot_filename,
                                            const char *png_filename) {
    if (!forest || !forest->trees || forest->n_trees <= 0 || !forest->trees[0].root) {
        printf("\n%s model is not trained yet. Train the model first.\n", model_name);
        return;
    }

    if (!export_decision_tree_to_dot(&forest->trees[0], data, dot_filename)) {
        printf("\nCould not save DOT file.\n");
        return;
    }

    printf("\nDOT visualization saved to: %s\n", dot_filename);

    if (export_decision_tree_to_png(&forest->trees[0], data, dot_filename, png_filename)) {
        printf("PNG visualization saved to: %s\n", png_filename);
        printf("Open this PNG file in VS Code to see the tree.\n");
    } else {
        printf("PNG file was not created automatically.\n");
        printf("Install Graphviz and make sure the 'dot' command is available in PATH.\n");
        printf("Then run this command manually:\n");
        printf("dot -Tpng %s -o %s\n", dot_filename, png_filename);
    }
}

/* Ввод сенсоров робота и получение двух прогнозов: команда + скорость. */
static void predict_robot_action(RandomForest *classification_forest,
                                 RandomForest *regression_forest,
                                 const Dataset *classification_data,
                                 const Dataset *regression_data) {
    if (!classification_forest->trees) {
        printf("\nClassification model is not trained yet. Training with Gini now...\n");
        train_classification(classification_forest, classification_data, CRITERION_GINI);
    }
    if (!regression_forest->trees) {
        printf("\nRegression model is not trained yet. Training now...\n");
        train_regression(regression_forest, regression_data);
    }

    double x[4];
    printf("\nEnter robot sensor values.\n");
    x[0] = read_double_value("Distance to obstacle, cm: ");
    x[1] = read_double_value("Current speed, m/s: ");
    x[2] = read_double_value("Obstacle angle, degrees [-90; 90]: ");
    x[3] = read_double_value("Battery charge, percent: ");

    int action = predict_forest_class(classification_forest, x);
    double safe_speed = predict_forest_regression(regression_forest, x);

    printf("\nModel prediction:\n");
    printf("Robot command: %s\n", classification_data->class_names[action]);
    printf("Recommended safe speed: %.3f m/s\n", safe_speed);
}

/* Главное меню консольной программы. */
static void print_menu(void) {
    printf("\n=============================================\n");
    printf(" DECISION TREE WITH BAGGING FOR ROBOT\n");
    printf("=============================================\n");
    printf("1. Train classification using Gini\n");
    printf("2. Train classification using Information Gain\n");
    printf("3. Train safe-speed regression\n");
    printf("4. Enter sensor values and predict\n");
    printf("5. Show first dataset rows\n");
    printf("6. Export classification tree visualization\n");
    printf("7. Export regression tree visualization\n");
    printf("0. Exit\n");
}

int main(void) {
    setlocale(LC_NUMERIC, "C");
    srand((unsigned int)time(NULL));

    /* Датасеты создаются автоматически, чтобы проект запускался без внешних зависимостей. */
    Dataset classification_data = create_robot_classification_dataset(240);
    Dataset regression_data = create_robot_regression_dataset(240);

    RandomForest classification_forest;
    classification_forest.trees = NULL;
    classification_forest.n_trees = 0;
    classification_forest.task = TASK_CLASSIFICATION;
    classification_forest.n_classes = classification_data.n_classes;

    RandomForest regression_forest;
    regression_forest.trees = NULL;
    regression_forest.n_trees = 0;
    regression_forest.task = TASK_REGRESSION;
    regression_forest.n_classes = 0;

    int running = 1;
    while (running) {
        print_menu();
        int choice = read_int_value("Select option: ");

        switch (choice) {
            case 1:
                train_classification(&classification_forest, &classification_data, CRITERION_GINI);
                break;
            case 2:
                train_classification(&classification_forest, &classification_data, CRITERION_INFO_GAIN);
                break;
            case 3:
                train_regression(&regression_forest, &regression_data);
                break;
            case 4:
                predict_robot_action(&classification_forest, &regression_forest,
                                     &classification_data, &regression_data);
                break;
            case 5:
                printf("\nClassification dataset:\n");
                print_dataset_head(&classification_data, TASK_CLASSIFICATION, 8);
                printf("\nRegression dataset:\n");
                print_dataset_head(&regression_data, TASK_REGRESSION, 8);
                break;
            case 6:
                if (!classification_forest.trees) {
                    printf("\nClassification model is not trained yet. Training with Gini now...\n");
                    train_classification(&classification_forest, &classification_data, CRITERION_GINI);
                }
                export_first_tree_visualization(&classification_forest, &classification_data,
                                                "Classification", "tree_classification.dot",
                                                "tree_classification.png");
                break;
            case 7:
                if (!regression_forest.trees) {
                    printf("\nRegression model is not trained yet. Training now...\n");
                    train_regression(&regression_forest, &regression_data);
                }
                export_first_tree_visualization(&regression_forest, &regression_data,
                                                "Regression", "tree_regression.dot",
                                                "tree_regression.png");
                break;
            case 0:
                running = 0;
                break;
            default:
                printf("Unknown option. Try again.\n");
                break;
        }
    }

    free_random_forest(&classification_forest);
    free_random_forest(&regression_forest);
    free_dataset(&classification_data);
    free_dataset(&regression_data);

    printf("Program finished.\n");
    return 0;
}
