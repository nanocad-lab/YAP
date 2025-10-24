#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon


class Die:
    def __init__(
        self, DIE_W_um, DIE_L_um, die_center, NUM_PADS_PER_DIE,
        DIE_VERTEX_COORDS, PAD_ARR_BOX, 
        pad_yield_flag: bool,
    ):
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.die_center = die_center
        self.num_pads = NUM_PADS_PER_DIE
        self.vertices_coords = self.get_vertices_coords(die_center, DIE_VERTEX_COORDS)
        self.pad_array_box = PAD_ARR_BOX + die_center

        self.survival = True
        self.voids_occur = False

        self.die_yield = {}
        self.pad_yield_map = {}
        self.pad_yield_flag = pad_yield_flag

    def get_vertices_coords(self, die_center, DIE_VERTEX_COORDS):
        vertices_coords = DIE_VERTEX_COORDS + die_center
        return vertices_coords
    



class Wafer:
    def __init__(
        self,
        wafer_radius: float,
        DIE_W_um: float,
        DIE_L_um: float,
        PAD_ARR_ROW: int,
        PAD_ARR_COL: int,
        PAD_TOP_R_um: float,
        PAD_BOT_R_um: float,
        base_pad_coords: np.ndarray,
        dice_width: float,
        pad_yield_flag: bool,
        dice_proportion=1.0,
    ):
        self.wafer_radius = wafer_radius
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.PAD_ARR_ROW = PAD_ARR_ROW
        self.PAD_ARR_COL = PAD_ARR_COL
        self.PAD_TOP_R_um = PAD_TOP_R_um
        self.PAD_BOT_R_um = PAD_BOT_R_um
        self.die_list = []
        self.num_dies = 0
        self.dice_proportion = dice_proportion
        self.voids = []
        self.safe_voids_mask = []
        self.roughness_voids = []
        self.survival_die = 0
        self.base_pad_coords = base_pad_coords
        self.dice_width = dice_width
        self.pad_yield_flag = pad_yield_flag
        self.glb_pad_yield_min_max_dict = {}

    def generate_die(self, NUM_PADS_PER_DIE, DIE_VERTEX_COORDS, PAD_ARR_BOX):
        die_row = 2 * self.wafer_radius // (self.DIE_L_um + self.dice_width) + 1
        die_col = 2 * self.wafer_radius // (self.DIE_W_um + self.dice_width) + 1
        flag_die_outside = False
        for i in range(int(die_row)):
            for j in range(int(die_col)):
                flag_die_outside = False
                die_center = np.array(
                    [
                        - die_col * (self.DIE_W_um + self.dice_width) / 2
                        + (self.DIE_W_um + self.dice_width) / 2
                        + j * (self.DIE_W_um + self.dice_width),
                        die_row * (self.DIE_L_um + self.dice_width) / 2
                        - (self.DIE_L_um + self.dice_width) / 2
                        - i * (self.DIE_L_um + self.dice_width),
                    ]
                )
                if (
                    np.sqrt(die_center[0] ** 2 + die_center[1] ** 2)
                    >= self.wafer_radius * self.dice_proportion
                ):
                    flag_die_outside = True
                    continue
                die = Die(
                    self.DIE_W_um,
                    self.DIE_L_um,
                    die_center,
                    NUM_PADS_PER_DIE,
                    DIE_VERTEX_COORDS,
                    PAD_ARR_BOX,
                    pad_yield_flag=self.pad_yield_flag
                )
                for vertex in die.vertices_coords:
                    if (
                        np.sqrt(vertex[0] ** 2 + vertex[1] ** 2)
                        >= self.wafer_radius * self.dice_proportion
                    ):
                        flag_die_outside = True
                        break
                if flag_die_outside:
                    continue
                self.die_list.append(die)
        self.num_dies = len(self.die_list)

    def draw_wafer_die(self, fig_size=(30, 30), draw_pad_yield_map_option=None):
        fig, ax = plt.subplots(figsize=fig_size, dpi=900)
        wafer_circle = plt.Circle((0, 0), self.wafer_radius, color="black", fill=False)
        ax.add_artist(wafer_circle)
        ax.set_xlim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        ax.set_ylim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        # draw dies
        for die in self.die_list:
            # Draw the pad array box
            polygon_coords = np.array([
                die.vertices_coords[0],  # top-left
                die.vertices_coords[1],  # top-right
                die.vertices_coords[3],  # bottom-right
                die.vertices_coords[2],  # bottom-left
            ])
            die_box = Polygon(polygon_coords, color="blue", fill=False)
            ax.add_patch(die_box)
            if die.survival == False:   # Draw a red edge if the die is not survived
                die_box = Polygon(polygon_coords, color="red", fill=False)
            elif die.voids_occur == True:
                die_box = Polygon(polygon_coords, color="green", fill=False)
            ax.add_patch(die_box)

            # Draw the overlay pad yield map
            if draw_pad_yield_map_option == 'Y_ovl':  # Draw the overlay yield map
                if hasattr(die, 'pad_yield_map') and "Y_ovl" in die.pad_yield_map:
                    overlay_yield_map = die.pad_yield_map['Y_ovl']
                    ax.imshow(
                        overlay_yield_map,
                        extent=[
                            die.vertices_coords[0][0], # x_min
                            die.vertices_coords[1][0], # x_max
                            die.vertices_coords[2][1], # y_min
                            die.vertices_coords[0][1]  # y_max
                        ],
                        origin='upper',
                        cmap='viridis',
                        vmin=self.glb_pad_yield_min_max_dict['Y_ovl'][0], # global min
                        vmax=self.glb_pad_yield_min_max_dict['Y_ovl'][1], # global max
                        alpha=0.5,
                    )
            # Draw the defect pad yield map
            if draw_pad_yield_map_option == 'Y_df':  # Draw the defect yield map
                if hasattr(die, 'pad_yield_map') and "Y_df" in die.pad_yield_map:
                    defect_yield_map = die.pad_yield_map['Y_df']
                    ax.imshow(
                        defect_yield_map,
                        extent=[
                            die.vertices_coords[0][0], # x_min
                            die.vertices_coords[1][0], # x_max
                            die.vertices_coords[2][1], # y_min
                            die.vertices_coords[0][1]  # y_max
                        ],
                        origin='upper',
                        cmap='viridis',
                        vmin=self.glb_pad_yield_min_max_dict['Y_df'][0], # global min
                        vmax=self.glb_pad_yield_min_max_dict['Y_df'][1], # global max
                        alpha=0.5,
                    )
            # draw pads
            die_pad_coords = die.die_center + self.base_pad_coords
            for pad in die_pad_coords:
                if pad[0] != np.nan and pad[1] != np.nan:   # There is a pad/bump
                    ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_um, color='darkorange', fill=True, alpha=1.0))
                    ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_um, color='lightgreen', fill=True, alpha=1.0))

        # Draw voids
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="black", fill=False))
        ax.set_aspect("equal")
        plt.show()
        # # Save the wafer figure
        fig.savefig("wafer_die.png")    





def wafer_initialize(
    NUM_WAFERS,
    DIE_W_um,
    DIE_L_um,
    PAD_ARR_W_um,
    PAD_ARR_L_um,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PITCH_r_um,
    PITCH_c_um,
    WAF_R_um,
    PAD_TOP_R_um,
    PAD_BOT_R_um,
    dice_width,
    pad_bitmap_collection,
    pad_yield_flag: bool = False,
):
    waf_list = []
    # Calculate the die center standard coordinates
    DIE_VERTEX_COORDS = np.array(
        [
            [-DIE_W_um / 2, DIE_L_um / 2],
            [DIE_W_um / 2, DIE_L_um / 2],
            [-DIE_W_um / 2, -DIE_L_um / 2],
            [DIE_W_um / 2, -DIE_L_um / 2],
        ]
    )  # die vertex coordinates: [top-left, top-right, bottom-left, bottom-right]
    PAD_ARR_BOX = np.array(
        [
            [-PAD_ARR_W_um / 2, PAD_ARR_L_um / 2], 
            [PAD_ARR_W_um / 2, PAD_ARR_L_um / 2], 
            [-PAD_ARR_W_um / 2, -PAD_ARR_L_um / 2], 
            [PAD_ARR_W_um / 2, -PAD_ARR_L_um / 2]])
    
    # Calculate the total number of pads per die
    NUM_PADS_PER_DIE = pad_bitmap_collection['num_critical_pads'] + pad_bitmap_collection['num_redundant_pads'] + pad_bitmap_collection['num_dummy_pads']
    

    if pad_bitmap_collection['pad_coords'] is not None:
        PAD_COORDS = pad_bitmap_collection['pad_coords']
    else:
        if PITCH_r_um >= 1.0 and PITCH_c_um >= 1.0:
            # Specify the pad coordinates based on pitch and array size
            PAD_COORDS = np.zeros([PAD_ARR_ROW * PAD_ARR_COL, 2], dtype=np.float32)  # pad coordinates: [x, y]
        
            # Create grid of row and column indices
            col_indices = np.arange(PAD_ARR_COL)
            row_indices = np.arange(PAD_ARR_ROW)
            col_grid, row_grid = np.meshgrid(col_indices, row_indices)

            # Calculate x and y coordinates
            x_coords = (-PAD_ARR_W_um / 2 + col_grid * PITCH_c_um).astype(np.float32)
            y_coords = (PAD_ARR_L_um / 2 - row_grid * PITCH_r_um).astype(np.float32)

            # Combine x and y coordinates
            PAD_COORDS = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
        else:
            print("Too many Cu pads... Will not generate the pad coordinates.")
            PAD_COORDS = None
    
    # Initialize the wafer
    for i in range(NUM_WAFERS):
        wafer = Wafer(
            wafer_radius=WAF_R_um,
            DIE_W_um=DIE_W_um,
            DIE_L_um=DIE_L_um,
            PAD_ARR_ROW=PAD_ARR_ROW,
            PAD_ARR_COL=PAD_ARR_COL,
            PAD_TOP_R_um=PAD_TOP_R_um,
            PAD_BOT_R_um=PAD_BOT_R_um,
            base_pad_coords=PAD_COORDS,
            dice_width=dice_width,
            pad_yield_flag=pad_yield_flag,
        )
        wafer.generate_die(NUM_PADS_PER_DIE, DIE_VERTEX_COORDS, PAD_ARR_BOX)
        # wafer.draw_wafer_die()
        # break
        wafer.survival_die = len(wafer.die_list)
        waf_list.append(wafer)
    # print("{} dies in the wafer.".format(len(wafer.die_list)))
    
    return waf_list



