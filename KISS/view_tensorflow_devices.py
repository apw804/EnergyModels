import tensorflow as tf

# get a list of available physical devices
physical_devices = tf.config.list_physical_devices()

# get detailed information about each device
for device in physical_devices:
    print(f"Device name: {device.name}")
    print(f"Device type: {device.device_type}")
    device_details = tf.config.experimental.get_device_details(device.name)
    print(f"Device details:\n{device_details}\n")