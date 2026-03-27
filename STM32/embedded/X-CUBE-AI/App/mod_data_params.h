/**
  ******************************************************************************
  * @file    mod_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-26T23:59:58+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef MOD_DATA_PARAMS_H
#define MOD_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_MOD_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_mod_data_weights_params[1]))
*/

#define AI_MOD_DATA_CONFIG               (NULL)


#define AI_MOD_DATA_ACTIVATIONS_SIZES \
  { 768, }
#define AI_MOD_DATA_ACTIVATIONS_SIZE     (768)
#define AI_MOD_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MOD_DATA_ACTIVATION_1_SIZE    (768)



#define AI_MOD_DATA_WEIGHTS_SIZES \
  { 69140, }
#define AI_MOD_DATA_WEIGHTS_SIZE         (69140)
#define AI_MOD_DATA_WEIGHTS_COUNT        (1)
#define AI_MOD_DATA_WEIGHT_1_SIZE        (69140)



#define AI_MOD_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_mod_activations_table[1])

extern ai_handle g_mod_activations_table[1 + 2];



#define AI_MOD_DATA_WEIGHTS_TABLE_GET() \
  (&g_mod_weights_table[1])

extern ai_handle g_mod_weights_table[1 + 2];


#endif    /* MOD_DATA_PARAMS_H */
