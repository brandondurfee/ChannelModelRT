import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.rt import Receiver, coverage_map
import numpy as np

class ChannelModelRT():
    r"""
    Class implementing a ray tracing channel model
    
    Simulates a channel between a single transmitter and a single receiver

    Example Usage
    -------------
    scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")
    scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")
    
    # ChannelModelRT channel model
    rt_channel = ChannelModelRT(scene, delay_spread, carrier_frequency, "uplink", min_speed)
    channel = OFDMChannel(rt_channel, resource_grid, normalize_channel = True, return_channel = True)


    Parameters
    -----------
    config : dictionary ({})
        config map with parameters to initialize ChannelModelRT
            Has parameters:
                scene : Scene
                coverage_map : Coverage_Map
                delay_spread, 
                direction, 
                carrier_frequency, 
                min_speed, 
                max_speed, 
                synthetic_array, 
                max_depth
                edge_diffraction
                num_samples
                max_gain_db
                min_gain_db
                min_dist
                max_dist
                subcarrier_spacing
                normalize_delays
                num_paths

        
    Input
    --------
    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps (number of OFDM symbols)

    sampling_frequency : float
        Sampling frequency [Hz] (subcarrier spacing)

    Output
    ---------
    a : [batch size, num_rx = 1, num_rx_ant, num_tx = 1, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self, config, dtype = tf.complex64):
        # load values in from config map to initialize class attributes

        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        real_dtype = dtype.real_dtype
        self._real_dtype = real_dtype

        # direction
        assert config["direction"] in('uplink', 'downlink'), "Invalid link direction"
        self._direction = config["direction"]
        
        # scene
        assert config["scene"] is not None, "scene must be initially loaded"
        self._scene = config["scene"]

        # coverage map
        assert coverage_map is not None, "coverage_map must be initially computed"
        self._coverage_map = config["coverage_map"]

        # direction
        if self._direction == 'downlink':
            self._moving_end = 'rx'
            self._tx_array = self._scene.tx_array
            self._rx_array = self._scene.rx_array
        elif self._direction == 'uplink':
            self._moving_end = 'tx'
            self._tx_array = self._scene.rx_array
            self._rx_array = self._scene.tx_array

        # carrier frequency
        self._carrier_frequency = tf.constant(config["carrier_frequency"], real_dtype)

        # delay spread
        self._delay_spread = tf.constant(config["delay_spread"], real_dtype)

        # min speed
        self._min_speed = tf.constant(config["min_speed"], real_dtype)

        # max speed
        if config["max_speed"] is None:
            self._max_speed = self._min_speed
        else:
            assert config["max_speed"] >= self._min_speed, \
                "min_speed cannot be larger than max_speed"
            self._max_speed = tf.constant(config["max_speed"], real_dtype)
        
        # synthetic array
        if config["synthetic_array"]:
            self._synthetic_array = config["synthetic_array"]
        else:
            self._synthetic_array = True

        # max depth
        if config["max_depth"]:
            self._max_depth = config["max_depth"]
        else:
            self._max_depth = 10
        
        # edge diffraction
        if config["edge_diffraction"]:
            self._edge_diffraction = config["edge_diffraction"]
        else:
            self._edge_diffraction = False
        
        # num samples
        if config["num_samples"]:
            self._num_samples = config["num_samples"]
        else:
            self._num_samples = 1e7

        # max gain db
        if config["max_gain_db"]:
            self._max_gain_db = config["max_gain_db"]
        else:
            self._max_gain_db = None
        
        # min gain db
        if config["min_gain_db"]:
            self._min_gain_db = config["min_gain_db"]
        else:
            self._min_gain_db = None
        
        # min dist
        self._min_dist = config["min_dist"]

        # max dist
        self._max_dist = config["max_dist"]

        # subcarrier spacing
        self._subcarrier_spacing = config["subcarrier_spacing"]

        # normalize delays
        if config["normalize_delays"] == True:
            self._normalize_delays = True
        else:
            self._normalize_delays = False
        
        # num paths
        assert config["num_paths"] is not None, "set num_paths in config"
        self._num_paths = config["num_paths"]


        
    def __call__(self, batch_size, num_time_steps, sampling_frequency):
        # 1. for each batch size -> randomly sample a position
        # 2. Add receivers to the scene at these random locations
        # 3. Simulate paths between transmitters and receivers
        # 4. Apply doppler shift based on random speeds
        # 5. calculate channel impulse response -> alpha, tau
        # 6. save data to file
        # 7. remove rx from scene
        # 7. return alpha, tau

        batch_size = int(batch_size)

        if (type(batch_size) != int or batch_size == None or batch_size == 0):
            batch_size = 1 # so we only create one sample_position

        # 1. for each batch size -> randomly sample a position for the transmitters
        # self._rdtype = self._dtype.real_dtype

        # min_val_db = -1 * np.infty
        # min_val_db = tf.constant(min_val_db, self._rdtype)

        # max_val_db = np.infty
        # max_val_db = tf.constant(max_val_db, self._rdtype)

        ue_pos = self._coverage_map.sample_positions(num_pos = int(batch_size),
                                                min_dist = self._min_dist,
                                                max_dist = self._max_dist)

        # ue_pos is of shape (1, 64, 3) -> get dim 1
        ue_pos = ue_pos[0].numpy()[0]

        # 2. Add receivers to the scene at these random locations
        for i in range(batch_size):
            self._scene.remove(f"rx-{i}") # remove receivers just in case:
            rx = Receiver(name = f"rx-{i}", position = ue_pos[i])
            self._scene.add(rx)

        # 3. Simulate paths between transmitters and receivers
        paths = self._scene.compute_paths(max_depth = self._max_depth, diffraction = True, edge_diffraction = self._edge_diffraction, num_samples = self._num_samples)

        # 4. apply doppler shifts based on random speeds
        paths.reverse_directions = True if self._direction == "uplink" else False
        paths.normalize_delays = self._normalize_delays   # False
        rx_vel = [np.random.uniform(self._min_speed, self._max_speed), np.random.uniform(self._min_speed, self._max_speed), 0]

        paths.apply_doppler(sampling_frequency=self._subcarrier_spacing,
                            num_time_steps=num_time_steps,
                            tx_velocities=[0.,0.,0],
                            rx_velocities=rx_vel)

        # 5. calculate channel impulse response -> alpha, tau
        h, delays = paths.cir(self._num_paths)

        del paths # free mem

        for i in range(batch_size):
            self._scene.remove(f"rx-{i}")

        # Reshaping to match the expected output
        h = tf.transpose(h, [1, 0, 2, 3, 4, 5, 6])
        delays = tf.transpose(delays, [1, 0, 2, 3])
        
        return h, delays
        