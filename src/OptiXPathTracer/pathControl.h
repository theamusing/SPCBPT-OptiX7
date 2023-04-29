#pragma once

#define PT_BRDF_STRATEGY_ONLY
//#define PT_NEE_STRATEGY_ONLY



/* 如果想要控制BDPT渲染哪些光路，需要打开这个开关 */
static const bool BDPT_CONTROL = 1;

/* 在PT中，打开这个只会渲染含有glossy顶点的路径 */
/* 在BDPT中，打开这个会在光子路选择是只选择焦散光子路，
    即末端是glossy顶点的光子路 */
static const bool S_ONLY = 1;

/* A 代表 Any*/

/* L - * - E*/
static const bool LAE_ENABLE = 0;

static const bool LE_ENABLE = 1;
static const bool LDE_ENABLE = 0;
static const bool LDSE_ENABLE = 0;


/* LS */
static const bool LS_ENABLE = 0;
static const bool LSE_ENABLE = 0;
static const bool LSDE_ENABLE = 0;

/* LDS */
static const bool LDS_ENABLE = 1;
static const bool LDSDE_ENABLE = 0;

/* LS(S)* */
static const bool LS_S_ENABLE = 0;
static const bool LS_SDE_ENABLE = 0;
static const bool LSSDE_ENABLE = 0;
static const bool LSSSDE_ENABLE = 0;

