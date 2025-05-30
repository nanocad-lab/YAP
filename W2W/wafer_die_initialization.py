#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm


class Die:
    def __init__(
        self, die_width, die_length, die_center, DIE_VERTEX_COORDS, PAD_COORDS, PAD_ARR_BOX
    ):
        self.die_width = die_width
        self.die_length = die_length
        self.die_center = die_center
        self.num_pad = len(PAD_COORDS)
        self.vertices_coords = self.get_vertices_coords(die_center, DIE_VERTEX_COORDS)
        self.pad_array_box = PAD_ARR_BOX + die_center
        self.avg_misalignment_far = 0  # average misalignment of pad
        self.survival = True

    def get_vertices_coords(self, die_center, DIE_VERTEX_COORDS):
        vertices_coords = DIE_VERTEX_COORDS + die_center
        return vertices_coords


class Wafer:
    def __init__(
        self,
        wafer_radius,
        die_width,
        die_length,
        pad_top_radius,
        pad_bot_radius,
        base_pad_coords,
        dice_width,
        dice_proportion=1.0,
    ):
        self.wafer_radius = wafer_radius
        self.die_width = die_width
        self.die_length = die_length
        self.pad_top_radius = pad_top_radius
        self.pad_bot_radius = pad_bot_radius
        self.die_list = []
        self.dice_proportion = dice_proportion
        self.voids = []
        self.safe_voids_mask = []
        self.roughness_voids = []
        self.survival_die = 0
        self.base_pad_coords = base_pad_coords
        self.dice_width = dice_width

    def generate_die(self, DIE_VERTEX_COORDS, PAD_COORDS, PAD_ARR_BOX):
        die_row = 2 * self.wafer_radius // (self.die_length + self.dice_width) + 1
        die_col = 2 * self.wafer_radius // (self.die_width + self.dice_width) + 1
        flag_die_outside = False
        for i in range(int(die_row)):
            for j in range(int(die_col)):
                flag_die_outside = False
                die_center = np.array(
                    [
                        -die_col * (self.die_width + self.dice_width) / 2
                        + (self.die_width + self.dice_width) / 2
                        + j * (self.die_width + self.dice_width),
                        die_row * (self.die_length + self.dice_width) / 2
                        - (self.die_length + self.dice_width) / 2
                        - i * (self.die_length + self.dice_width),
                    ]
                )
                if (
                    np.sqrt(die_center[0] ** 2 + die_center[1] ** 2)
                    >= self.wafer_radius * self.dice_proportion
                ):
                    flag_die_outside = True
                    continue
                die = Die(
                    self.die_width,
                    self.die_length,
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
        fig, ax = plt.subplots(figsize=fig_size, dpi=900)
        wafer_circle = plt.Circle((0, 0), self.wafer_radius, color="black", fill=False)
        ax.add_artist(wafer_circle)
        ax.set_xlim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        ax.set_ylim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        # draw dies
        for die in self.die_list:
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
            if die.survival == True:
                # ax.plot(
                #     [die.vertices_coords[0][0], die.vertices_coords[1][0]],
                #     [die.vertices_coords[0][1], die.vertices_coords[1][1]],
                #     color="blue",
                # )
                # ax.plot(
                #     [die.vertices_coords[1][0], die.vertices_coords[3][0]],
                #     [die.vertices_coords[1][1], die.vertices_coords[3][1]],
                #     color="blue",
                # )
                # ax.plot(
                #     [die.vertices_coords[2][0], die.vertices_coords[3][0]],
                #     [die.vertices_coords[2][1], die.vertices_coords[3][1]],
                #     color="blue",
                # )
                # ax.plot(
                #     [die.vertices_coords[2][0], die.vertices_coords[0][0]],
                #     [die.vertices_coords[2][1], die.vertices_coords[0][1]],
                #     color="blue",
                # )
                _ = 1
            else:   # Draw a red cross if the die is not survived
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
                # ax.plot(
                #     [die.vertices_coords[0][0], die.vertices_coords[1][0]],
                #     [die.vertices_coords[0][1], die.vertices_coords[1][1]],
                #     color="black",
                # )
                # ax.plot(
                #     [die.vertices_coords[1][0], die.vertices_coords[3][0]],
                #     [die.vertices_coords[1][1], die.vertices_coords[3][1]],
                #     color="black",
                # )
                # ax.plot(
                #     [die.vertices_coords[2][0], die.vertices_coords[3][0]],
                #     [die.vertices_coords[2][1], die.vertices_coords[3][1]],
                #     color="black",
                # )
                # ax.plot(
                #     [die.vertices_coords[2][0], die.vertices_coords[0][0]],
                #     [die.vertices_coords[2][1], die.vertices_coords[0][1]],
                #     color="black",
                # )
            # draw pads
            # die_pad_coords = die.center + PAD_COORDS
            # for pad in die_pad_coords:
            #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.pad_top_radius, color='blue', fill=False))
            #     ax.add_artist(patches.Circle((pad[0], pad[1]), self.pad_bot_radius, color='orange', fill=False))

        # Draw voids
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="black", fill=False))
        ax.set_aspect("equal")
        plt.show()
        # # Save the wafer figure
        fig.savefig("wafer_die.png")    


def wafer_initialize(
    NUM_WAFERS,
    DIE_W,
    DIE_L,
    PAD_ARR_W,
    PAD_ARR_L,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PITCH,
    WAF_R,
    PAD_TOP_R,
    PAD_BOT_R,
    dice_width
):
    waf_list = []
    # Calculate the die center standard coordinates
    DIE_VERTEX_COORDS = np.array(
        [
            [-DIE_W / 2, DIE_L / 2],
            [DIE_W / 2, DIE_L / 2],
            [-DIE_W / 2, -DIE_L / 2],
            [DIE_W / 2, -DIE_L / 2],
        ]
    )  # die vertex coordinates: [top-left, top-right, bottom-left, bottom-right]
    PAD_ARR_BOX = np.array(
        [
            [-PAD_ARR_W / 2, PAD_ARR_L / 2], 
            [PAD_ARR_W / 2, PAD_ARR_L / 2], 
            [-PAD_ARR_W / 2, -PAD_ARR_L / 2], 
            [PAD_ARR_W / 2, -PAD_ARR_L / 2]])

    # Calculate the top-left pad coordinates of the pad array
    PAD_COORDS = np.zeros([PAD_ARR_ROW * PAD_ARR_COL, 2])
    # for row in range(PAD_ARR_ROW):
    #     for col in range(PAD_ARR_COL):
    #         PAD_COORDS[row * PAD_ARR_COL + col] = np.array(
    #             [-PAD_ARR_W / 2 + col * PITCH, PAD_ARR_L / 2 - row * PITCH]
    #         )
    # Create grid of row and column indices
    col_indices = np.arange(PAD_ARR_COL)
    row_indices = np.arange(PAD_ARR_ROW)
    col_grid, row_grid = np.meshgrid(col_indices, row_indices)

    # Calculate x and y coordinates
    x_coords = -PAD_ARR_W / 2 + col_grid * PITCH
    y_coords = PAD_ARR_L / 2 - row_grid * PITCH

    # Combine x and y coordinates
    PAD_COORDS = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)

    # Initialize the wafer
    for i in range(NUM_WAFERS):
        wafer = Wafer(
            wafer_radius=WAF_R,
            die_width=DIE_W,
            die_length=DIE_L,
            pad_top_radius=PAD_TOP_R,
            pad_bot_radius=PAD_BOT_R,
            base_pad_coords=PAD_COORDS,
            dice_width=dice_width
        )
        wafer.generate_die(DIE_VERTEX_COORDS, PAD_COORDS, PAD_ARR_BOX)
        # wafer.draw_wafer_die()
        # break
        wafer.survival_die = len(wafer.die_list)
        waf_list.append(wafer)
    # print("{} dies in the wafer.".format(len(wafer.die_list)))
    return waf_list