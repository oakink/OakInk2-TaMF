import os
import numpy as np
import torch


class ActionRecognitionAdapter(torch.utils.data.Dataset):

    def __init__(self, interaction_segment_dataset):
        super().__init__()

        self.interaction_segment_dataset = interaction_segment_dataset
        self.action_list = [
            'cap', 'scoop', 'pour', 'wipe', 'spread', 'grip', 'scrape', 'rearrange', 'press_button', 'place_onto',
            'take_outside', 'hold', 'cut', 'screw', 'assemble', 'stir', 'unscrew', 'trigger_lever', 'open_gate',
            'place_inside', 'close_gate', 'uncap', 'brush_whiteboard', 'close_laptop_lid', 'use_keyboard', 'remove_usb',
            'remove_power_plug', 'plug_in_power_plug', 'insert_usb', 'use_gamecontroller', 'insert_lightbulb',
            'pull_out_drawer', 'insert_pencil', 'sharpen_pencil', 'remove_pencil', 'write_on_paper', 'remove_lid',
            'put_on_lid', 'shear_paper', 'staple_paper_together', 'remove_the_pen_cap', 'write_on_whiteboard',
            'cap_the_pen', 'put_flower_into_vase', 'push_in_drawer', 'remove_lightbulb', 'open_laptop_lid', 'open_book',
            'use_mouse', 'remove_from_test_tube_rack', 'hold_test_tube', 'heat_test_tube',
            'place_test_tube_on_rack_with_holder', 'pour_in_lab', 'place_on_test_tube_rack', 'put_off_alcohol_lamp',
            'shake_lab_container', 'place_asbestos_mesh', 'uncap_alcohol_lamp', 'ignite_alcohol_lamp', 'heat_beaker',
            'stir_experiment_substances', 'remove_test_tube', 'swap', 'remove_test_tube_from_rack_with_holder',
            'flip_open_tooth_paste_cap', 'squeeze_tooth_paste', 'flip_close_tooth_paste_cap', 'close_book'
        ]
        self.max_action = len(self.action_list)

    def __getitem__(self, index):
        sample = self.interaction_segment_dataset[index]

        action_label = str(sample["info"][1].split(':')[0])
        action_label_id = self.action_list.index(action_label)
        # create onehot vector of self.max_action
        action_onehot = np.zeros(self.max_action, dtype=np.int32)
        action_onehot[action_label_id] = 1

        sample["action_label"] = action_label
        sample["action_label_id"] = action_label_id
        sample["action_onehot"] = action_onehot
        return sample

    def __len__(self):
        return len(self.interaction_segment_dataset)
