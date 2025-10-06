#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Die:
    def __init__(
        self, DIE_W_umidth, DIE_L_umength, die_center, NUM_PADS_PER_DIE,
        DIE_VERTEX_COORDS, PAD_ARR_BOX, 
        pad_boundary_bitmap_coords
    ):
        self.DIE_W_umidth = DIE_W_umidth
        self.DIE_L_umength = DIE_L_umength
        self.die_center = die_center
        self.num_pad = NUM_PADS_PER_DIE
        self.vertices_coords = self.get_vertices_coords(die_center, DIE_VERTEX_COORDS)
        self.pad_array_box = PAD_ARR_BOX + die_center
        # Calculate the outermost critical pad and redundant pad coordinates based on the pad boundary bitmap index
        self.ovl_critical_pad_boundary_coords = pad_boundary_bitmap_coords + die_center

        self.survival = True
        self.voids_occur = False

    def get_vertices_coords(self, die_center, DIE_VERTEX_COORDS):
        vertices_coords = DIE_VERTEX_COORDS + die_center
        return vertices_coords
    



class Wafer:
    def __init__(
        self,
        wafer_radius,
        DIE_W_umidth,
        DIE_L_umength,
        PAD_TOP_R_umadius,
        PAD_BOT_R_umadius,
        base_pad_coords,
        dice_width,
        die_pad_PITCH_um,
        dice_proportion=1.0,
    ):
        self.wafer_radius = wafer_radius
        self.DIE_W_umidth = DIE_W_umidth
        self.DIE_L_umength = DIE_L_umength
        self.PAD_TOP_R_umadius = PAD_TOP_R_umadius
        self.PAD_BOT_R_umadius = PAD_BOT_R_umadius
        self.DIE_L_umist = []
        self.dice_proportion = dice_proportion
        self.voids = []
        self.safe_voids_mask = []
        self.roughness_voids = []
        self.survival_die = 0
        self.base_pad_coords = base_pad_coords
        self.dice_width = dice_width
        self.die_pad_PITCH_um = die_pad_PITCH_um

    def generate_die(self, NUM_PADS_PER_DIE, DIE_VERTEX_COORDS, PAD_ARR_BOX, pad_boundary_bitmap_coords):
        die_row = 2 * self.wafer_radius // (self.DIE_L_umength + self.dice_width) + 1
        die_col = 2 * self.wafer_radius // (self.DIE_W_umidth + self.dice_width) + 1
        flag_die_outside = False
        for i in range(int(die_row)):
            for j in range(int(die_col)):
                flag_die_outside = False
                die_center = np.array(
                    [
                        -die_col * (self.DIE_W_umidth + self.dice_width) / 2
                        + (self.DIE_W_umidth + self.dice_width) / 2
                        + j * (self.DIE_W_umidth + self.dice_width),
                        die_row * (self.DIE_L_umength + self.dice_width) / 2
                        - (self.DIE_L_umength + self.dice_width) / 2
                        - i * (self.DIE_L_umength + self.dice_width),
                    ]
                )
                if (
                    np.sqrt(die_center[0] ** 2 + die_center[1] ** 2)
                    >= self.wafer_radius * self.dice_proportion
                ):
                    flag_die_outside = True
                    continue
                die = Die(
                    self.DIE_W_umidth,
                    self.DIE_L_umength,
                    die_center,
                    NUM_PADS_PER_DIE,
                    DIE_VERTEX_COORDS,
                    PAD_ARR_BOX,
                    pad_boundary_bitmap_coords,
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
                self.DIE_L_umist.append(die)

    def draw_wafer_die(self, fig_size=(30, 30)):
        fig, ax = plt.subplots(figsize=fig_size, dpi=900)
        wafer_circle = plt.Circle((0, 0), self.wafer_radius, color="black", fill=False)
        ax.add_artist(wafer_circle)
        ax.set_xlim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        ax.set_ylim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        # draw dies
        for die in self.DIE_L_umist:
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
            if die.survival == False:   # Draw a red edge if the die is not survived
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[1][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[1][1]],
                    color="red",
                )
                ax.plot(
                    [die.vertices_coords[1][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[1][1], die.vertices_coords[3][1]],
                    color="red",
                )
                ax.plot(
                    [die.vertices_coords[2][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[2][1], die.vertices_coords[3][1]],
                    color="red",
                )
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[2][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[2][1]],
                    color="red",
                )
            elif die.voids_occur == True:
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[1][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[1][1]],
                    color="green",
                )
                ax.plot(
                    [die.vertices_coords[1][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[1][1], die.vertices_coords[3][1]],
                    color="green",
                )
                ax.plot(
                    [die.vertices_coords[2][0], die.vertices_coords[3][0]],
                    [die.vertices_coords[2][1], die.vertices_coords[3][1]],
                    color="green",
                )
                ax.plot(
                    [die.vertices_coords[0][0], die.vertices_coords[2][0]],
                    [die.vertices_coords[0][1], die.vertices_coords[2][1]],
                    color="green",
                )
            # draw pads
            # die_pad_coords = die.center + PAD_COORDS
            # for pad in die_pad_coords:
            #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_umadius, color='blue', fill=False))
            #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_umadius, color='orange', fill=False))

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
    PITCH_um,
    WAF_R_um,
    PAD_TOP_R_um,
    PAD_BOT_R_um,
    dice_width,
    pad_bitmap_collection
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
    
    NUM_PADS_PER_DIE = PAD_ARR_ROW * PAD_ARR_COL  # number of pads in a die
    
    # Calculate the top-left pad coordinates of the pad array
    if PITCH_um >= 1.0:
        PAD_COORDS = np.zeros([PAD_ARR_ROW * PAD_ARR_COL, 2], dtype=np.float16)  # pad coordinates: [x, y]
    
        # Create grid of row and column indices
        col_indices = np.arange(PAD_ARR_COL)
        row_indices = np.arange(PAD_ARR_ROW)
        col_grid, row_grid = np.meshgrid(col_indices, row_indices)

        # Calculate x and y coordinates
        x_coords = (-PAD_ARR_W_um / 2 + col_grid * PITCH_um).astype(np.float16)
        y_coords = (PAD_ARR_L_um / 2 - row_grid * PITCH_um).astype(np.float16)

        # Combine x and y coordinates
        PAD_COORDS = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
    else:
        print("Too many Cu pads... Will not generate the pad coordinates.")
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
        redundant_pad_boundary_bitmap_coords = redundant_copy_pad_boundary_bitmap_row_col_block_ind * pad_block_size * PITCH_um + [-PAD_ARR_W_um / 2, PAD_ARR_L_um / 2]
        pad_boundary_bitmap_coords = np.concatenate((critical_pad_boundary_bitmap_coords, redundant_pad_boundary_bitmap_coords), axis=0)
    else:
        pad_boundary_bitmap_coords = critical_pad_boundary_bitmap_coords
    

    # Initialize the wafer
    for i in range(NUM_WAFERS):
        wafer = Wafer(
            wafer_radius=WAF_R_um,
            DIE_W_umidth=DIE_W_um,
            DIE_L_umength=DIE_L_um,
            PAD_TOP_R_umadius=PAD_TOP_R_um,
            PAD_BOT_R_umadius=PAD_BOT_R_um,
            base_pad_coords=PAD_COORDS,
            dice_width=dice_width,
            die_pad_PITCH_um=PITCH_um,
        )
        wafer.generate_die(NUM_PADS_PER_DIE, DIE_VERTEX_COORDS, PAD_ARR_BOX, pad_boundary_bitmap_coords)
        # wafer.draw_wafer_die()
        # break
        wafer.survival_die = len(wafer.DIE_L_umist)
        waf_list.append(wafer)
    # print("{} dies in the wafer.".format(len(wafer.DIE_L_umist)))
    
    return waf_list



