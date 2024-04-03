import carla

constant_name = "ClearNoon"
constant = getattr(carla.WeatherParameters, constant_name)
print(constant)
