from data_parser import batch_convert
from consumption_forecast import generate_forecast
from consumption_forecast_comparison import plot_actual_vs_forecast_with_energy

def main(input_dir, output_dir):
    batch_convert(input_dir, output_dir)
    generate_forecast(output_dir)
    plot_actual_vs_forecast_with_energy(output_dir)

if __name__ == "__main__":
    for day in ["mixed", "sunny", "very-sunny", "overcast"]:
        main(f"data/{day}", f"out/{day}")
