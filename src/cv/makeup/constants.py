

# Vars
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5

# LANDMARKS
LOWER_LIP_INNER     = [308, 324, 318,402, 317, 14, 87, 178, 88, 95, 78]
LOWER_lIP_OUTER     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

UPPER_LIP_OUTER     = [61, 185, 40, 39, 37, 11, 267, 269, 270, 409, 291]
UPPER_LIP_INEER     = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

LEFT_EYEBROW_LOWER  = [276, 283, 282, 295, 285]
RIGHT_EYEBROW_LOWER = [46, 53, 52, 65, 55]

LEFT_EYE_UPPER      = [362, 398, 384, 385, 386, 387, 388, 466, 263]
RIGHT_EYE_UPPER     = [133, 173, 157, 158, 159, 160, 161, 246, 33]

LEFT_EYELINER       = [353,  260,  385,386, 387, 388, 466, 263, 249]
RIGHT_EYELINER      = [124, 30, 158, 159, 160, 161, 246, 33, 7 ]

LEFT_CHEEK_BONE     = [227,50, 205, 206, 92,210, 138, 215, 177 ]
RIGHT_CHEEK_BONE    = [447, 280, 425, 426, 322, 430, 367, 435 , 401]

lEFT_CHEEK          = [227,34, 111, 100, 36, 205, 187, 147, 137]
RIGHT_CHEEK         = [447, 264, 340, 329, 266, 425, 411, 376, 366 ]

UPPER_FACE          = [356, 389, 251, 332, 297,338,10,109, 67,103,54,21,162,127, 34, 143,156,70,63,105,66, 107, 9,336,296,334,293,300,383,372,264]
LOWER_FACE          = [264, 447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135, 138, 215, 177, 137, 227, 34, 143, 111, 118, 120,  128, 122, 6 ,   351, 357, 349,  347,340 ,372 ]
FACE                = [356, 389, 251, 332, 297,338,10,109, 67,103,54,21,162,127, 234, 93, 132,58, 172, 136, 150 , 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454 ]


# POLYGONS
LOWER_LIP           = LOWER_lIP_OUTER + LOWER_LIP_INNER
UPPER_LIP           = UPPER_LIP_OUTER + UPPER_LIP_INEER
LEFT_EYESHADOW      = LEFT_EYE_UPPER + LEFT_EYEBROW_LOWER
RIGHT_EYESHADOW     = RIGHT_EYE_UPPER + RIGHT_EYEBROW_LOWER
FACE                = UPPER_FACE + LOWER_FACE
LIPS = LOWER_LIP+UPPER_LIP

# MAKEUPS
LIPS                = [LIPS]
EYESHADOW           = [LEFT_EYESHADOW, RIGHT_EYESHADOW]
EYELINER            = [LEFT_EYELINER, RIGHT_EYELINER]
CONCEALER           = [LEFT_CHEEK_BONE, RIGHT_CHEEK_BONE]
FOUNDATION          = [FACE]
BLUSH               = [lEFT_CHEEK, RIGHT_CHEEK]