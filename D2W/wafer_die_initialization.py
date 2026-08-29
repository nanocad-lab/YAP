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
        self, DIE_W_um, DIE_L_um, die_center, 
        DIE_VERTEX_COORDS, num_pads, PAD_ARR_BOX,
        pad_boundary_bitmap_coords,
        pad_yield_flag: bool,
        BASE_PAD_COORDS: np.ndarray = None,
    ):
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.die_center = die_center
        self.num_pads = num_pads
        self.vertices_coords = self.get_vertices_coords(die_center, DIE_VERTEX_COORDS)
        self.pad_array_box = PAD_ARR_BOX + die_center
        self.ovl_critical_pad_boundary_coords = pad_boundary_bitmap_coords + die_center
        self.pad_array = BASE_PAD_COORDS + die_center if pad_yield_flag == True else None

        self.survival = True
        self.safe_voids_mask = []
        self.voids = []
        self.voids_occur = False

        self.die_yield = {}
        self.pad_yield_map = {}
        self.glb_pad_yield_min_max_dict = {}

    def get_vertices_coords(self, die_center, DIE_VERTEX_COORDS):
        vertices_coords = DIE_VERTEX_COORDS + die_center
        return vertices_coords
    
    def draw_die(self, ax):
        # Draw the pad array box
        polygon_coords = np.array([
            self.pad_array_box[0],  # top-left
            self.pad_array_box[1],  # top-right
            self.pad_array_box[3],  # bottom-right
            self.pad_array_box[2],  # bottom-left
        ])
        die_box = Polygon(polygon_coords, color="blue", fill=False)
        ax.add_patch(die_box)
        # draw die outline
        if self.survival == False:
            die_box = Polygon(self.vertices_coords, color="red", fill=False)
        elif self.voids_occur == True:
            die_box = Polygon(self.vertices_coords, color="green", fill=False)
        ax.add_patch(die_box)
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="red", fill=False))
        ax.set_aspect("equal")
        # set x and y axis limits
        ax.set_xlim(-self.DIE_W_um*0.6, self.DIE_W_um*0.6)
        ax.set_ylim(-self.DIE_L_um*0.6, self.DIE_L_um*0.6)
        plt.show()

        # # draw pads
        # for pad in die.pad_coords:
        #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_um, color='blue', fill=False))
        #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_um, color='orange', fill=False))    


class Wafer:
    def __init__(
        self,
        wafer_radius,
        DIE_W_um,
        DIE_L_um,
        PAD_TOP_R_um,
        PAD_BOT_R_um,
        base_pad_coords,
        dice_width,
        dice_proportion=1.0,
    ):
        self.wafer_radius = wafer_radius
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.PAD_TOP_R_um = PAD_TOP_R_um
        self.PAD_BOT_R_um = PAD_BOT_R_um
        self.die_list = []
        self.dice_proportion = dice_proportion
        self.voids = []
        self.safe_voids_mask = []
        self.roughness_voids = []
        self.survival_die = 0
        self.base_pad_coords = base_pad_coords
        self.dice_width = dice_width

    def generate_die(self, DIE_VERTEX_COORDS, PAD_COORDS, PAD_ARR_BOX):
        die_row = 2 * self.wafer_radius // (self.DIE_L_um + self.dice_width) + 1
        die_col = 2 * self.wafer_radius // (self.DIE_W_um + self.dice_width) + 1
        flag_die_outside = False
        for i in range(int(die_row)):
            for j in range(int(die_col)):
                flag_die_outside = False
                die_center = np.array(
                    [
                        -die_col * (self.DIE_W_um + self.dice_width) / 2
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
                    DIE_VERTEX_COORDS,
                    PAD_COORDS,
                    PAD_ARR_BOX
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

    def draw_wafer_die(self, fig_size=(30, 30)):
        fig, ax = plt.subplots(figsize=fig_size)
        wafer_circle = plt.Circle((0, 0), self.wafer_radius, color="black", fill=False)
        ax.add_artist(wafer_circle)
        ax.set_xlim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        ax.set_ylim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        # draw dies
        for die in self.die_list:
            if die.survival == True:
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[1][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[1][1]],
                    color="black",
                )
                ax.plot(
                    [die.vertices_coords[1][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[1][1], die.vertices_coords[3][1]],
                    color="black",
                )
                ax.plot(
                    [die.vertices_coords[2][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[2][1], die.vertices_coords[3][1]],
                    color="black",
                )
                ax.plot(
                    [die.vertices_coords[2][0], die.vertices_coords[0][0]],
                    [die.vertices_coords[2][1], die.vertices_coords[0][1]],
                    color="black",
                )
            else:   # Draw a red cross if the die is not survived
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[3][1]],
                    color="red",
                )
                ax.plot(
                    [die.vertices_coords[1][0], die.vertices_coords[2][0]],
                    [die.vertices_coords[1][1], die.vertices_coords[2][1]],
                    color="red",
                )
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[1][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[1][1]],
                    color="black",
                )
                ax.plot(
                    [die.vertices_coords[1][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[1][1], die.vertices_coords[3][1]],
                    color="black",
                )
                ax.plot(
                    [die.vertices_coords[2][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[2][1], die.vertices_coords[3][1]],
                    color="black",
                )
                ax.plot(
                    [die.vertices_coords[2][0], die.vertices_coords[0][0]],
                    [die.vertices_coords[2][1], die.vertices_coords[0][1]],
                    color="black",
                )
            # Draw the pad array box
            ax.plot(
                [die.pad_array_box[0][0], die.pad_array_box[1][0]],
                [die.pad_array_box[0][1], die.pad_array_box[1][1]],
                color="blue",
            )
            ax.plot(
                [die.pad_array_box[1][0], die.pad_array_box[3][0]],
                [die.pad_array_box[1][1], die.pad_array_box[3][1]],
                color="blue",
            )
            ax.plot(
                [die.pad_array_box[2][0], die.pad_array_box[3][0]],
                [die.pad_array_box[2][1], die.pad_array_box[3][1]],
                color="blue",
            )
            ax.plot(
                [die.pad_array_box[2][0], die.pad_array_box[0][0]],
                [die.pad_array_box[2][1], die.pad_array_box[0][1]],
                color="blue",
            )
            # draw pads
            # die_pad_coords = die.center + PAD_COORDS
            # for pad in die_pad_coords:
            #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_um, color='blue', fill=False))
            #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_um, color='orange', fill=False))

        # Draw voids
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="red", fill=False))
        ax.set_aspect("equal")
        plt.show()
        # # Save the wafer figure
        # fig.savefig("wafer_die.png")    


def die_initialize(
    NUM_DIES,
    DIE_W_um,
    DIE_L_um,
    PAD_ARR_W_um,
    PAD_ARR_L_um,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PITCH_um,
    pad_bitmap_collection,
    pad_yield_flag: bool = False,
    generate_pad_coords: bool = True,
):
    die_list = []
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

    num_pads = PAD_ARR_ROW * PAD_ARR_COL  # Total number of pads in the pad array

    if generate_pad_coords and PITCH_um >= 1.0:
    # Calculate the top-left pad coordinates of the pad array
        PAD_COORDS = np.zeros([PAD_ARR_ROW * PAD_ARR_COL, 2], dtype=np.float32)  # pad coordinates: [x, y]

        # Create grid of row and column indices
        col_indices = np.arange(PAD_ARR_COL)
        row_indices = np.arange(PAD_ARR_ROW)
        col_grid, row_grid = np.meshgrid(col_indices, row_indices)

        # Calculate x and y coordinates
        x_coords = (-PAD_ARR_W_um / 2 + col_grid * PITCH_um).astype(np.float32)
        y_coords = (PAD_ARR_L_um / 2 - row_grid * PITCH_um).astype(np.float32)

        # Combine x and y coordinates
        PAD_COORDS = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
    elif generate_pad_coords:
        print("Too many Cu pads... Will not generate the pad coordinates.")
        PAD_COORDS = None
    else:
        PAD_COORDS = None

    # Get the outer coordinates of the critical pads
    # (row, col) of the critical pads, top left corner, top right corner, bottom left corner, bottom right corner
    pad_block_size = pad_bitmap_collection["pad_block_size"]
    critical_pad_boundary_bitmap_row_col_block_ind = pad_bitmap_collection["critical_pad_boundary_bitmap_row_col_block_ind"] 
    critical_pad_boundary_bitmap_row_col_block_ind_non_zero_mask = (critical_pad_boundary_bitmap_row_col_block_ind != 0).astype(int)
    # We did some fine tuning here to make sure the coordinates are correct
    origin = [-PAD_ARR_W_um / 2, -PAD_ARR_L_um / 2]
    bias = critical_pad_boundary_bitmap_row_col_block_ind * pad_block_size * PITCH_um - critical_pad_boundary_bitmap_row_col_block_ind_non_zero_mask * [(DIE_W_um - PAD_ARR_W_um), (DIE_L_um - PAD_ARR_L_um)]
    critical_pad_boundary_bitmap_coords = bias + origin
    redundant_copy_pad_boundary_bitmap_row_col_block_ind = pad_bitmap_collection["redundant_copy_pad_boundary_bitmap_row_col_block_ind"]
    # If there are redundant pads, concatenate their coordinates and critical pad coordinates as the pad boundary coordinates (considered in the overlahy error)
    if redundant_copy_pad_boundary_bitmap_row_col_block_ind is not None:
        redundant_pad_boundary_bitmap_coords = redundant_copy_pad_boundary_bitmap_row_col_block_ind * pad_block_size * PITCH_um + [-PAD_ARR_W_um / 2, -PAD_ARR_L_um / 2]
        pad_boundary_bitmap_coords = np.concatenate((critical_pad_boundary_bitmap_coords, redundant_pad_boundary_bitmap_coords), axis=0)
    else:
        pad_boundary_bitmap_coords = critical_pad_boundary_bitmap_coords
        
    for i in range(NUM_DIES):
        die = Die(
            DIE_W_um=DIE_W_um,
            DIE_L_um=DIE_L_um,
            die_center=np.array([0, 0]),
            DIE_VERTEX_COORDS=DIE_VERTEX_COORDS,
            num_pads=num_pads,
            PAD_ARR_BOX=PAD_ARR_BOX,
            pad_boundary_bitmap_coords=pad_boundary_bitmap_coords,
            pad_yield_flag=pad_yield_flag,
            BASE_PAD_COORDS=PAD_COORDS,
        )
        die_list.append(die)
    return die_list, PAD_COORDS
