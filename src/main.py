from evaluate import evaluate_model
from train import train_model
from utils import *


def main():
    # Define parameters for different configurations
    configs = [
        {"input_size": 8, "hidden_size": 3, "output_size": 8, "task": "8-3-8", "data_path": "data/encoder_decoder_8_3_8_training_data.txt"},
        {"input_size": 8, "hidden_size": 2, "output_size": 8, "task": "8-2-8", "data_path": "data/encoder_decoder_8_2_8_training_data.txt"},
        {"input_size": 50, "hidden_size": 2, "output_size": 50, "task": "50-2-50", "data_path": "data/encoder_decoder_50_2_50_training_data.txt"},
        {"input_size": 64, "hidden_size": 6, "output_size": 64, "task": "64-6-64", "data_path": "data/encoder_decoder_64_6_64_training_data.txt"}
    ]
    learning_rate = 0.005
    epochs = 10000

    # Training and evaluation for each configuration
    for config in configs:
        data_path = config['data_path']
        log_path = "logs"
        plot_path = "reports"

        input_size, output_size, task_name = config["input_size"], config["output_size"], config["task"]
        num_samples, train_data, test_data = generate_training_data(num_samples=1000, n_bits=input_size,
                                                                    task_name=task_name)

        print(f"\nTraining for {config['task']} with data {data_path}")
        model, hidden_outputs, losses = train_model(train_data, config["input_size"], config["hidden_size"],
                                                    config["output_size"], learning_rate, epochs)

        # Save the learning curve and logs
        plot_title = f'Learning Curve for {task_name} Encoder-Decoder \nwith Logistic Function and Learning Rate = {learning_rate}'
        plot_learning_curve(losses, f'{plot_path}/learning_curve_{task_name}.png', plot_title)
        with open(f'{log_path}/learning_curve_{task_name}.txt', 'w') as f:
            for loss in losses:
                f.write(f"{loss}\n")

        print(f"Evaluating for {config['task']}")
        predictions, hidden_states, mse = evaluate_model(model, test_data, input_size, output_size)

        # print("Hidden layer states for all patterns:")
        # print(hidden_outputs)

        print_results(test_data, predictions, f'{plot_path}/output_{task_name}.txt')

        print(f"Mean Squared Error (MSE): {mse}")
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------\n")

        visualize_hidden_states(hidden_states, int(task_name.split('-')[1]), f'{plot_path}/hidden_states_{task_name}.png')


if __name__ == "__main__":
    main()
