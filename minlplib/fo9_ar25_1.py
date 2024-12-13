# MINLP written by GAMS Convert at 02/17/22 17:19:36
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#       435        1        2      432        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#       180      108        0       72        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#      1734     1716       18
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.i1 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i2 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i3 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i4 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i5 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i6 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i7 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i8 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i9 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i10 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i11 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i12 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i13 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i14 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i15 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i16 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i17 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i18 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i19 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i20 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i21 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i22 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i23 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i24 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i25 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i26 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i27 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i28 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i29 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i30 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i31 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i32 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i33 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i34 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i35 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i36 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i37 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i38 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i39 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i40 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i41 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i42 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i43 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i44 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i45 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i46 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i47 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i48 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i49 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i50 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i51 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i52 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i53 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i54 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i55 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i56 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i57 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i58 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i59 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i60 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i61 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i62 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i63 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i64 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i65 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i66 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i67 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i68 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i69 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i70 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i71 = Var(within=Integers, bounds=(0,100), initialize=0)
m.i72 = Var(within=Integers, bounds=(0,100), initialize=0)
m.x73 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x74 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x75 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x76 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x77 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x78 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x79 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x80 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x81 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x82 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x83 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x84 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x85 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x86 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x87 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x88 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x89 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x90 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x91 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x92 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x93 = Var(within=Reals, bounds=(2.5298,6.3246), initialize=2.5298)
m.x94 = Var(within=Reals, bounds=(2.5298,6.3246), initialize=2.5298)
m.x95 = Var(within=Reals, bounds=(2.5298,6.3246), initialize=2.5298)
m.x96 = Var(within=Reals, bounds=(2.5298,6.3246), initialize=2.5298)
m.x97 = Var(within=Reals, bounds=(2.5298,6.3246), initialize=2.5298)
m.x98 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x99 = Var(within=Reals, bounds=(2.5298,6.3246), initialize=2.5298)
m.x100 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x101 = Var(within=Reals, bounds=(3.7947,9.4868), initialize=3.7947)
m.x102 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x103 = Var(within=Reals, bounds=(3.7947,9.4868), initialize=3.7947)
m.x104 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x105 = Var(within=Reals, bounds=(3.7947,9.4868), initialize=3.7947)
m.x106 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x107 = Var(within=Reals, bounds=(3.7947,9.4868), initialize=3.7947)
m.x108 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x109 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x110 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x111 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x112 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x113 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x114 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x115 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x116 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x117 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x118 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x119 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x120 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x121 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x122 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x123 = Var(within=Reals, bounds=(1.8974,4.7434), initialize=1.8974)
m.x124 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x125 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x126 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x127 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x128 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x129 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x130 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x131 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x132 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x133 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x134 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x135 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x136 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x137 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x138 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x139 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x140 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x141 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x142 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x143 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x144 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x145 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x146 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x147 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x148 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x149 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x150 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x151 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x152 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x153 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x154 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x155 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x156 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x157 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x158 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x159 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x160 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x161 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x162 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x163 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x164 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x165 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x166 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x167 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x168 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x169 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x170 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x171 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x172 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x173 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x174 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x175 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x176 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x177 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x178 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x179 = Var(within=Reals, bounds=(None,None), initialize=0)
m.x180 = Var(within=Reals, bounds=(None,None), initialize=0)

m.obj = Objective(sense=minimize, expr= m.x73 + m.x74 + m.x75 + m.x76 + m.x77
    + m.x78 + m.x79 + m.x80 + m.x81 + m.x82 + m.x83 + m.x84 + m.x85 + m.x86 +
    m.x87 + m.x88)

m.e1 = Constraint(expr= m.x89 - m.x90 >= 0)
m.e2 = Constraint(expr= m.x91 - m.x92 >= 0)
m.e3 = Constraint(expr= m.i1 - m.i2 == 0)
m.e4 = Constraint(expr= -12 * m.i1 - m.x73 + 0.5 * m.x93 + 0.5 * m.x94 <= 0)
m.e5 = Constraint(expr= 13 * m.i1 - m.x74 + 0.5 * m.x95 + 0.5 * m.x96 <= 13)
m.e6 = Constraint(expr= -12 * m.i3 + 0.5 * m.x93 + 0.5 * m.x97 - m.x98 <= 0)
m.e7 = Constraint(expr= 13 * m.i3 + 0.5 * m.x95 + 0.5 * m.x99 - m.x100 <= 13)
m.e8 = Constraint(expr= -12 * m.i5 + 0.5 * m.x93 + 0.5 * m.x101 - m.x102 <= 0)
m.e9 = Constraint(expr= 13 * m.i5 + 0.5 * m.x95 + 0.5 * m.x103 - m.x104 <= 13)
m.e10 = Constraint(expr= -12 * m.i7 + 0.5 * m.x93 + 0.5 * m.x105 - m.x106 <= 0)
m.e11 = Constraint(expr= 13 * m.i7 + 0.5 * m.x95 + 0.5 * m.x107 - m.x108 <= 13)
m.e12 = Constraint(expr= -12 * m.i9 + 0.5 * m.x93 + 0.5 * m.x109 - m.x110 <= 0)
m.e13 = Constraint(expr= 13 * m.i9 + 0.5 * m.x95 + 0.5 * m.x111 - m.x112 <= 13)
m.e14 = Constraint(expr= -12 * m.i11 + 0.5 * m.x93 + 0.5 * m.x113 - m.x114
    <= 0)
m.e15 = Constraint(expr= 13 * m.i11 + 0.5 * m.x95 + 0.5 * m.x115 - m.x116
    <= 13)
m.e16 = Constraint(expr= -12 * m.i13 + 0.5 * m.x93 + 0.5 * m.x117 - m.x118
    <= 0)
m.e17 = Constraint(expr= 13 * m.i13 + 0.5 * m.x95 + 0.5 * m.x119 - m.x120
    <= 13)
m.e18 = Constraint(expr= -12 * m.i15 + 0.5 * m.x93 + 0.5 * m.x121 - m.x122
    <= 0)
m.e19 = Constraint(expr= 13 * m.i15 + 0.5 * m.x95 + 0.5 * m.x123 - m.x124
    <= 13)
m.e20 = Constraint(expr= -12 * m.i17 - m.x75 + 0.5 * m.x94 + 0.5 * m.x97 <= 0)
m.e21 = Constraint(expr= 13 * m.i17 - m.x76 + 0.5 * m.x96 + 0.5 * m.x99 <= 13)
m.e22 = Constraint(expr= -12 * m.i19 + 0.5 * m.x94 + 0.5 * m.x101 - m.x125
    <= 0)
m.e23 = Constraint(expr= 13 * m.i19 + 0.5 * m.x96 + 0.5 * m.x103 - m.x126
    <= 13)
m.e24 = Constraint(expr= -12 * m.i21 + 0.5 * m.x94 + 0.5 * m.x105 - m.x127
    <= 0)
m.e25 = Constraint(expr= 13 * m.i21 + 0.5 * m.x96 + 0.5 * m.x107 - m.x128
    <= 13)
m.e26 = Constraint(expr= -12 * m.i23 + 0.5 * m.x94 + 0.5 * m.x109 - m.x129
    <= 0)
m.e27 = Constraint(expr= 13 * m.i23 + 0.5 * m.x96 + 0.5 * m.x111 - m.x130
    <= 13)
m.e28 = Constraint(expr= -12 * m.i25 + 0.5 * m.x94 + 0.5 * m.x113 - m.x131
    <= 0)
m.e29 = Constraint(expr= 13 * m.i25 + 0.5 * m.x96 + 0.5 * m.x115 - m.x132
    <= 13)
m.e30 = Constraint(expr= -12 * m.i27 + 0.5 * m.x94 + 0.5 * m.x117 - m.x133
    <= 0)
m.e31 = Constraint(expr= 13 * m.i27 + 0.5 * m.x96 + 0.5 * m.x119 - m.x134
    <= 13)
m.e32 = Constraint(expr= -12 * m.i29 + 0.5 * m.x94 + 0.5 * m.x121 - m.x135
    <= 0)
m.e33 = Constraint(expr= 13 * m.i29 + 0.5 * m.x96 + 0.5 * m.x123 - m.x136
    <= 13)
m.e34 = Constraint(expr= -12 * m.i31 - m.x77 + 0.5 * m.x97 + 0.5 * m.x101 <= 0)
m.e35 = Constraint(expr= 13 * m.i31 - m.x78 + 0.5 * m.x99 + 0.5 * m.x103 <= 13)
m.e36 = Constraint(expr= -12 * m.i33 + 0.5 * m.x97 + 0.5 * m.x105 - m.x137
    <= 0)
m.e37 = Constraint(expr= 13 * m.i33 + 0.5 * m.x99 + 0.5 * m.x107 - m.x138
    <= 13)
m.e38 = Constraint(expr= -12 * m.i35 + 0.5 * m.x97 + 0.5 * m.x109 - m.x139
    <= 0)
m.e39 = Constraint(expr= 13 * m.i35 + 0.5 * m.x99 + 0.5 * m.x111 - m.x140
    <= 13)
m.e40 = Constraint(expr= -12 * m.i37 + 0.5 * m.x97 + 0.5 * m.x113 - m.x141
    <= 0)
m.e41 = Constraint(expr= 13 * m.i37 + 0.5 * m.x99 + 0.5 * m.x115 - m.x142
    <= 13)
m.e42 = Constraint(expr= -12 * m.i39 + 0.5 * m.x97 + 0.5 * m.x117 - m.x143
    <= 0)
m.e43 = Constraint(expr= 13 * m.i39 + 0.5 * m.x99 + 0.5 * m.x119 - m.x144
    <= 13)
m.e44 = Constraint(expr= -12 * m.i41 + 0.5 * m.x97 + 0.5 * m.x121 - m.x145
    <= 0)
m.e45 = Constraint(expr= 13 * m.i41 + 0.5 * m.x99 + 0.5 * m.x123 - m.x146
    <= 13)
m.e46 = Constraint(expr= -12 * m.i43 - m.x79 + 0.5 * m.x101 + 0.5 * m.x105
    <= 0)
m.e47 = Constraint(expr= 13 * m.i43 - m.x80 + 0.5 * m.x103 + 0.5 * m.x107
    <= 13)
m.e48 = Constraint(expr= -12 * m.i45 + 0.5 * m.x101 + 0.5 * m.x109 - m.x147
    <= 0)
m.e49 = Constraint(expr= 13 * m.i45 + 0.5 * m.x103 + 0.5 * m.x111 - m.x148
    <= 13)
m.e50 = Constraint(expr= -12 * m.i47 + 0.5 * m.x101 + 0.5 * m.x113 - m.x149
    <= 0)
m.e51 = Constraint(expr= 13 * m.i47 + 0.5 * m.x103 + 0.5 * m.x115 - m.x150
    <= 13)
m.e52 = Constraint(expr= -12 * m.i49 + 0.5 * m.x101 + 0.5 * m.x117 - m.x151
    <= 0)
m.e53 = Constraint(expr= 13 * m.i49 + 0.5 * m.x103 + 0.5 * m.x119 - m.x152
    <= 13)
m.e54 = Constraint(expr= -12 * m.i51 + 0.5 * m.x101 + 0.5 * m.x121 - m.x153
    <= 0)
m.e55 = Constraint(expr= 13 * m.i51 + 0.5 * m.x103 + 0.5 * m.x123 - m.x154
    <= 13)
m.e56 = Constraint(expr= -12 * m.i53 - m.x81 + 0.5 * m.x105 + 0.5 * m.x109
    <= 0)
m.e57 = Constraint(expr= 13 * m.i53 - m.x82 + 0.5 * m.x107 + 0.5 * m.x111
    <= 13)
m.e58 = Constraint(expr= -12 * m.i55 + 0.5 * m.x105 + 0.5 * m.x113 - m.x155
    <= 0)
m.e59 = Constraint(expr= 13 * m.i55 + 0.5 * m.x107 + 0.5 * m.x115 - m.x156
    <= 13)
m.e60 = Constraint(expr= -12 * m.i57 + 0.5 * m.x105 + 0.5 * m.x117 - m.x157
    <= 0)
m.e61 = Constraint(expr= 13 * m.i57 + 0.5 * m.x107 + 0.5 * m.x119 - m.x158
    <= 13)
m.e62 = Constraint(expr= -12 * m.i59 + 0.5 * m.x105 + 0.5 * m.x121 - m.x159
    <= 0)
m.e63 = Constraint(expr= 13 * m.i59 + 0.5 * m.x107 + 0.5 * m.x123 - m.x160
    <= 13)
m.e64 = Constraint(expr= -12 * m.i61 - m.x83 + 0.5 * m.x109 + 0.5 * m.x113
    <= 0)
m.e65 = Constraint(expr= 13 * m.i61 - m.x84 + 0.5 * m.x111 + 0.5 * m.x115
    <= 13)
m.e66 = Constraint(expr= -12 * m.i63 + 0.5 * m.x109 + 0.5 * m.x117 - m.x161
    <= 0)
m.e67 = Constraint(expr= 13 * m.i63 + 0.5 * m.x111 + 0.5 * m.x119 - m.x162
    <= 13)
m.e68 = Constraint(expr= -12 * m.i65 + 0.5 * m.x109 + 0.5 * m.x121 - m.x163
    <= 0)
m.e69 = Constraint(expr= 13 * m.i65 + 0.5 * m.x111 + 0.5 * m.x123 - m.x164
    <= 13)
m.e70 = Constraint(expr= -12 * m.i67 - m.x85 + 0.5 * m.x113 + 0.5 * m.x117
    <= 0)
m.e71 = Constraint(expr= 13 * m.i67 - m.x86 + 0.5 * m.x115 + 0.5 * m.x119
    <= 13)
m.e72 = Constraint(expr= -12 * m.i69 + 0.5 * m.x113 + 0.5 * m.x121 - m.x165
    <= 0)
m.e73 = Constraint(expr= 13 * m.i69 + 0.5 * m.x115 + 0.5 * m.x123 - m.x166
    <= 13)
m.e74 = Constraint(expr= -12 * m.i71 - m.x87 + 0.5 * m.x117 + 0.5 * m.x121
    <= 0)
m.e75 = Constraint(expr= 13 * m.i71 - m.x88 + 0.5 * m.x119 + 0.5 * m.x123
    <= 13)
m.e76 = Constraint(expr= -0.395288 * m.x93 - 0.158112 * m.x95 <= -2)
m.e77 = Constraint(expr= -0.158113 * m.x93 - 0.395288 * m.x95 <= -2)
m.e78 = Constraint(expr= -0.395288 * m.x94 - 0.158112 * m.x96 <= -2)
m.e79 = Constraint(expr= -0.158113 * m.x94 - 0.395288 * m.x96 <= -2)
m.e80 = Constraint(expr= -0.395288 * m.x97 - 0.158112 * m.x99 <= -2)
m.e81 = Constraint(expr= -0.158113 * m.x97 - 0.395288 * m.x99 <= -2)
m.e82 = Constraint(expr= -0.263525 * m.x101 - 0.105408 * m.x103 <= -2)
m.e83 = Constraint(expr= -0.10541 * m.x101 - 0.263522 * m.x103 <= -2)
m.e84 = Constraint(expr= -0.263525 * m.x105 - 0.105408 * m.x107 <= -2)
m.e85 = Constraint(expr= -0.10541 * m.x105 - 0.263522 * m.x107 <= -2)
m.e86 = Constraint(expr= -0.527037 * m.x109 - 0.210822 * m.x111 <= -2)
m.e87 = Constraint(expr= -0.210819 * m.x109 - 0.527044 * m.x111 <= -2)
m.e88 = Constraint(expr= -0.527037 * m.x113 - 0.210822 * m.x115 <= -2)
m.e89 = Constraint(expr= -0.210819 * m.x113 - 0.527044 * m.x115 <= -2)
m.e90 = Constraint(expr= -0.527037 * m.x117 - 0.210822 * m.x119 <= -2)
m.e91 = Constraint(expr= -0.210819 * m.x117 - 0.527044 * m.x119 <= -2)
m.e92 = Constraint(expr= -0.527037 * m.x121 - 0.210822 * m.x123 <= -2)
m.e93 = Constraint(expr= -0.210819 * m.x121 - 0.527044 * m.x123 <= -2)
m.e94 = Constraint(expr= m.x89 + 0.5 * m.x93 <= 12)
m.e95 = Constraint(expr= -m.x89 + 0.5 * m.x93 <= 0)
m.e96 = Constraint(expr= m.x92 + 0.5 * m.x95 <= 13)
m.e97 = Constraint(expr= -m.x92 + 0.5 * m.x95 <= 0)
m.e98 = Constraint(expr= m.x90 + 0.5 * m.x94 <= 12)
m.e99 = Constraint(expr= -m.x90 + 0.5 * m.x94 <= 0)
m.e100 = Constraint(expr= m.x91 + 0.5 * m.x96 <= 13)
m.e101 = Constraint(expr= -m.x91 + 0.5 * m.x96 <= 0)
m.e102 = Constraint(expr= 0.5 * m.x97 + m.x167 <= 12)
m.e103 = Constraint(expr= 0.5 * m.x97 - m.x167 <= 0)
m.e104 = Constraint(expr= 0.5 * m.x99 + m.x168 <= 13)
m.e105 = Constraint(expr= 0.5 * m.x99 - m.x168 <= 0)
m.e106 = Constraint(expr= 0.5 * m.x101 + m.x169 <= 12)
m.e107 = Constraint(expr= 0.5 * m.x101 - m.x169 <= 0)
m.e108 = Constraint(expr= 0.5 * m.x103 + m.x170 <= 13)
m.e109 = Constraint(expr= 0.5 * m.x103 - m.x170 <= 0)
m.e110 = Constraint(expr= 0.5 * m.x105 + m.x171 <= 12)
m.e111 = Constraint(expr= 0.5 * m.x105 - m.x171 <= 0)
m.e112 = Constraint(expr= 0.5 * m.x107 + m.x172 <= 13)
m.e113 = Constraint(expr= 0.5 * m.x107 - m.x172 <= 0)
m.e114 = Constraint(expr= 0.5 * m.x109 + m.x173 <= 12)
m.e115 = Constraint(expr= 0.5 * m.x109 - m.x173 <= 0)
m.e116 = Constraint(expr= 0.5 * m.x111 + m.x174 <= 13)
m.e117 = Constraint(expr= 0.5 * m.x111 - m.x174 <= 0)
m.e118 = Constraint(expr= 0.5 * m.x113 + m.x175 <= 12)
m.e119 = Constraint(expr= 0.5 * m.x113 - m.x175 <= 0)
m.e120 = Constraint(expr= 0.5 * m.x115 + m.x176 <= 13)
m.e121 = Constraint(expr= 0.5 * m.x115 - m.x176 <= 0)
m.e122 = Constraint(expr= 0.5 * m.x117 + m.x177 <= 12)
m.e123 = Constraint(expr= 0.5 * m.x117 - m.x177 <= 0)
m.e124 = Constraint(expr= 0.5 * m.x119 + m.x178 <= 13)
m.e125 = Constraint(expr= 0.5 * m.x119 - m.x178 <= 0)
m.e126 = Constraint(expr= 0.5 * m.x121 + m.x179 <= 12)
m.e127 = Constraint(expr= 0.5 * m.x121 - m.x179 <= 0)
m.e128 = Constraint(expr= 0.5 * m.x123 + m.x180 <= 13)
m.e129 = Constraint(expr= 0.5 * m.x123 - m.x180 <= 0)
m.e130 = Constraint(expr= -m.x73 + m.x89 - m.x90 <= 0)
m.e131 = Constraint(expr= -m.x73 - m.x89 + m.x90 <= 0)
m.e132 = Constraint(expr= -m.x74 - m.x91 + m.x92 <= 0)
m.e133 = Constraint(expr= -m.x74 + m.x91 - m.x92 <= 0)
m.e134 = Constraint(expr= -12 * m.i1 - 12 * m.i2 - m.x89 + m.x90 + 0.5 * m.x93
    + 0.5 * m.x94 <= 0)
m.e135 = Constraint(expr= -12 * m.i1 + 12 * m.i2 + m.x89 - m.x90 + 0.5 * m.x93
    + 0.5 * m.x94 <= 12)
m.e136 = Constraint(expr= 13 * m.i1 - 13 * m.i2 + m.x91 - m.x92 + 0.5 * m.x95
    + 0.5 * m.x96 <= 13)
m.e137 = Constraint(expr= 13 * m.i1 + 13 * m.i2 - m.x91 + m.x92 + 0.5 * m.x95
    + 0.5 * m.x96 <= 26)
m.e138 = Constraint(expr= m.x89 - m.x98 - m.x167 <= 0)
m.e139 = Constraint(expr= -m.x89 - m.x98 + m.x167 <= 0)
m.e140 = Constraint(expr= m.x92 - m.x100 - m.x168 <= 0)
m.e141 = Constraint(expr= -m.x92 - m.x100 + m.x168 <= 0)
m.e142 = Constraint(expr= -12 * m.i3 - 12 * m.i4 - m.x89 + 0.5 * m.x93 + 0.5 *
    m.x97 + m.x167 <= 0)
m.e143 = Constraint(expr= -12 * m.i3 + 12 * m.i4 + m.x89 + 0.5 * m.x93 + 0.5 *
    m.x97 - m.x167 <= 12)
m.e144 = Constraint(expr= 13 * m.i3 - 13 * m.i4 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x99 + m.x168 <= 13)
m.e145 = Constraint(expr= 13 * m.i3 + 13 * m.i4 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x99 - m.x168 <= 26)
m.e146 = Constraint(expr= m.x89 - m.x102 - m.x169 <= 0)
m.e147 = Constraint(expr= -m.x89 - m.x102 + m.x169 <= 0)
m.e148 = Constraint(expr= m.x92 - m.x104 - m.x170 <= 0)
m.e149 = Constraint(expr= -m.x92 - m.x104 + m.x170 <= 0)
m.e150 = Constraint(expr= -12 * m.i5 - 12 * m.i6 - m.x89 + 0.5 * m.x93 + 0.5 *
    m.x101 + m.x169 <= 0)
m.e151 = Constraint(expr= -12 * m.i5 + 12 * m.i6 + m.x89 + 0.5 * m.x93 + 0.5 *
    m.x101 - m.x169 <= 12)
m.e152 = Constraint(expr= 13 * m.i5 - 13 * m.i6 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x103 + m.x170 <= 13)
m.e153 = Constraint(expr= 13 * m.i5 + 13 * m.i6 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x103 - m.x170 <= 26)
m.e154 = Constraint(expr= m.x89 - m.x106 - m.x171 <= 0)
m.e155 = Constraint(expr= -m.x89 - m.x106 + m.x171 <= 0)
m.e156 = Constraint(expr= m.x92 - m.x108 - m.x172 <= 0)
m.e157 = Constraint(expr= -m.x92 - m.x108 + m.x172 <= 0)
m.e158 = Constraint(expr= -12 * m.i7 - 12 * m.i8 - m.x89 + 0.5 * m.x93 + 0.5 *
    m.x105 + m.x171 <= 0)
m.e159 = Constraint(expr= -12 * m.i7 + 12 * m.i8 + m.x89 + 0.5 * m.x93 + 0.5 *
    m.x105 - m.x171 <= 12)
m.e160 = Constraint(expr= 13 * m.i7 - 13 * m.i8 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x107 + m.x172 <= 13)
m.e161 = Constraint(expr= 13 * m.i7 + 13 * m.i8 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x107 - m.x172 <= 26)
m.e162 = Constraint(expr= m.x89 - m.x110 - m.x173 <= 0)
m.e163 = Constraint(expr= -m.x89 - m.x110 + m.x173 <= 0)
m.e164 = Constraint(expr= m.x92 - m.x112 - m.x174 <= 0)
m.e165 = Constraint(expr= -m.x92 - m.x112 + m.x174 <= 0)
m.e166 = Constraint(expr= -12 * m.i9 - 12 * m.i10 - m.x89 + 0.5 * m.x93 + 0.5 *
    m.x109 + m.x173 <= 0)
m.e167 = Constraint(expr= -12 * m.i9 + 12 * m.i10 + m.x89 + 0.5 * m.x93 + 0.5 *
    m.x109 - m.x173 <= 12)
m.e168 = Constraint(expr= 13 * m.i9 - 13 * m.i10 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x111 + m.x174 <= 13)
m.e169 = Constraint(expr= 13 * m.i9 + 13 * m.i10 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x111 - m.x174 <= 26)
m.e170 = Constraint(expr= m.x89 - m.x114 - m.x175 <= 0)
m.e171 = Constraint(expr= -m.x89 - m.x114 + m.x175 <= 0)
m.e172 = Constraint(expr= m.x92 - m.x116 - m.x176 <= 0)
m.e173 = Constraint(expr= -m.x92 - m.x116 + m.x176 <= 0)
m.e174 = Constraint(expr= -12 * m.i11 - 12 * m.i12 - m.x89 + 0.5 * m.x93 + 0.5
    * m.x113 + m.x175 <= 0)
m.e175 = Constraint(expr= -12 * m.i11 + 12 * m.i12 + m.x89 + 0.5 * m.x93 + 0.5
    * m.x113 - m.x175 <= 12)
m.e176 = Constraint(expr= 13 * m.i11 - 13 * m.i12 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x115 + m.x176 <= 13)
m.e177 = Constraint(expr= 13 * m.i11 + 13 * m.i12 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x115 - m.x176 <= 26)
m.e178 = Constraint(expr= m.x89 - m.x118 - m.x177 <= 0)
m.e179 = Constraint(expr= -m.x89 - m.x118 + m.x177 <= 0)
m.e180 = Constraint(expr= m.x92 - m.x120 - m.x178 <= 0)
m.e181 = Constraint(expr= -m.x92 - m.x120 + m.x178 <= 0)
m.e182 = Constraint(expr= -12 * m.i13 - 12 * m.i14 - m.x89 + 0.5 * m.x93 + 0.5
    * m.x117 + m.x177 <= 0)
m.e183 = Constraint(expr= -12 * m.i13 + 12 * m.i14 + m.x89 + 0.5 * m.x93 + 0.5
    * m.x117 - m.x177 <= 12)
m.e184 = Constraint(expr= 13 * m.i13 - 13 * m.i14 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x119 + m.x178 <= 13)
m.e185 = Constraint(expr= 13 * m.i13 + 13 * m.i14 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x119 - m.x178 <= 26)
m.e186 = Constraint(expr= m.x89 - m.x122 - m.x179 <= 0)
m.e187 = Constraint(expr= -m.x89 - m.x122 + m.x179 <= 0)
m.e188 = Constraint(expr= m.x92 - m.x124 - m.x180 <= 0)
m.e189 = Constraint(expr= -m.x92 - m.x124 + m.x180 <= 0)
m.e190 = Constraint(expr= -12 * m.i15 - 12 * m.i16 - m.x89 + 0.5 * m.x93 + 0.5
    * m.x121 + m.x179 <= 0)
m.e191 = Constraint(expr= -12 * m.i15 + 12 * m.i16 + m.x89 + 0.5 * m.x93 + 0.5
    * m.x121 - m.x179 <= 12)
m.e192 = Constraint(expr= 13 * m.i15 - 13 * m.i16 - m.x92 + 0.5 * m.x95 + 0.5 *
    m.x123 + m.x180 <= 13)
m.e193 = Constraint(expr= 13 * m.i15 + 13 * m.i16 + m.x92 + 0.5 * m.x95 + 0.5 *
    m.x123 - m.x180 <= 26)
m.e194 = Constraint(expr= -m.x75 + m.x90 - m.x167 <= 0)
m.e195 = Constraint(expr= -m.x75 - m.x90 + m.x167 <= 0)
m.e196 = Constraint(expr= -m.x76 + m.x91 - m.x168 <= 0)
m.e197 = Constraint(expr= -m.x76 - m.x91 + m.x168 <= 0)
m.e198 = Constraint(expr= -12 * m.i17 - 12 * m.i18 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x97 + m.x167 <= 0)
m.e199 = Constraint(expr= -12 * m.i17 + 12 * m.i18 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x97 - m.x167 <= 12)
m.e200 = Constraint(expr= 13 * m.i17 - 13 * m.i18 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x99 + m.x168 <= 13)
m.e201 = Constraint(expr= 13 * m.i17 + 13 * m.i18 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x99 - m.x168 <= 26)
m.e202 = Constraint(expr= m.x90 - m.x125 - m.x169 <= 0)
m.e203 = Constraint(expr= -m.x90 - m.x125 + m.x169 <= 0)
m.e204 = Constraint(expr= m.x91 - m.x126 - m.x170 <= 0)
m.e205 = Constraint(expr= -m.x91 - m.x126 + m.x170 <= 0)
m.e206 = Constraint(expr= -12 * m.i19 - 12 * m.i20 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x101 + m.x169 <= 0)
m.e207 = Constraint(expr= -12 * m.i19 + 12 * m.i20 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x101 - m.x169 <= 12)
m.e208 = Constraint(expr= 13 * m.i19 - 13 * m.i20 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x103 + m.x170 <= 13)
m.e209 = Constraint(expr= 13 * m.i19 + 13 * m.i20 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x103 - m.x170 <= 26)
m.e210 = Constraint(expr= m.x90 - m.x127 - m.x171 <= 0)
m.e211 = Constraint(expr= -m.x90 - m.x127 + m.x171 <= 0)
m.e212 = Constraint(expr= m.x91 - m.x128 - m.x172 <= 0)
m.e213 = Constraint(expr= -m.x91 - m.x128 + m.x172 <= 0)
m.e214 = Constraint(expr= -12 * m.i21 - 12 * m.i22 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x105 + m.x171 <= 0)
m.e215 = Constraint(expr= -12 * m.i21 + 12 * m.i22 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x105 - m.x171 <= 12)
m.e216 = Constraint(expr= 13 * m.i21 - 13 * m.i22 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x107 + m.x172 <= 13)
m.e217 = Constraint(expr= 13 * m.i21 + 13 * m.i22 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x107 - m.x172 <= 26)
m.e218 = Constraint(expr= m.x90 - m.x129 - m.x173 <= 0)
m.e219 = Constraint(expr= -m.x90 - m.x129 + m.x173 <= 0)
m.e220 = Constraint(expr= m.x91 - m.x130 - m.x174 <= 0)
m.e221 = Constraint(expr= -m.x91 - m.x130 + m.x174 <= 0)
m.e222 = Constraint(expr= -12 * m.i23 - 12 * m.i24 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x109 + m.x173 <= 0)
m.e223 = Constraint(expr= -12 * m.i23 + 12 * m.i24 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x109 - m.x173 <= 12)
m.e224 = Constraint(expr= 13 * m.i23 - 13 * m.i24 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x111 + m.x174 <= 13)
m.e225 = Constraint(expr= 13 * m.i23 + 13 * m.i24 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x111 - m.x174 <= 26)
m.e226 = Constraint(expr= m.x90 - m.x131 - m.x175 <= 0)
m.e227 = Constraint(expr= -m.x90 - m.x131 + m.x175 <= 0)
m.e228 = Constraint(expr= m.x91 - m.x132 - m.x176 <= 0)
m.e229 = Constraint(expr= -m.x91 - m.x132 + m.x176 <= 0)
m.e230 = Constraint(expr= -12 * m.i25 - 12 * m.i26 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x113 + m.x175 <= 0)
m.e231 = Constraint(expr= -12 * m.i25 + 12 * m.i26 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x113 - m.x175 <= 12)
m.e232 = Constraint(expr= 13 * m.i25 - 13 * m.i26 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x115 + m.x176 <= 13)
m.e233 = Constraint(expr= 13 * m.i25 + 13 * m.i26 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x115 - m.x176 <= 26)
m.e234 = Constraint(expr= m.x90 - m.x133 - m.x177 <= 0)
m.e235 = Constraint(expr= -m.x90 - m.x133 + m.x177 <= 0)
m.e236 = Constraint(expr= m.x91 - m.x134 - m.x178 <= 0)
m.e237 = Constraint(expr= -m.x91 - m.x134 + m.x178 <= 0)
m.e238 = Constraint(expr= -12 * m.i27 - 12 * m.i28 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x117 + m.x177 <= 0)
m.e239 = Constraint(expr= -12 * m.i27 + 12 * m.i28 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x117 - m.x177 <= 12)
m.e240 = Constraint(expr= 13 * m.i27 - 13 * m.i28 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x119 + m.x178 <= 13)
m.e241 = Constraint(expr= 13 * m.i27 + 13 * m.i28 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x119 - m.x178 <= 26)
m.e242 = Constraint(expr= m.x90 - m.x135 - m.x179 <= 0)
m.e243 = Constraint(expr= -m.x90 - m.x135 + m.x179 <= 0)
m.e244 = Constraint(expr= m.x91 - m.x136 - m.x180 <= 0)
m.e245 = Constraint(expr= -m.x91 - m.x136 + m.x180 <= 0)
m.e246 = Constraint(expr= -12 * m.i29 - 12 * m.i30 - m.x90 + 0.5 * m.x94 + 0.5
    * m.x121 + m.x179 <= 0)
m.e247 = Constraint(expr= -12 * m.i29 + 12 * m.i30 + m.x90 + 0.5 * m.x94 + 0.5
    * m.x121 - m.x179 <= 12)
m.e248 = Constraint(expr= 13 * m.i29 - 13 * m.i30 - m.x91 + 0.5 * m.x96 + 0.5 *
    m.x123 + m.x180 <= 13)
m.e249 = Constraint(expr= 13 * m.i29 + 13 * m.i30 + m.x91 + 0.5 * m.x96 + 0.5 *
    m.x123 - m.x180 <= 26)
m.e250 = Constraint(expr= -m.x77 + m.x167 - m.x169 <= 0)
m.e251 = Constraint(expr= -m.x77 - m.x167 + m.x169 <= 0)
m.e252 = Constraint(expr= -m.x78 + m.x168 - m.x170 <= 0)
m.e253 = Constraint(expr= -m.x78 - m.x168 + m.x170 <= 0)
m.e254 = Constraint(expr= -12 * m.i31 - 12 * m.i32 + 0.5 * m.x97 + 0.5 * m.x101
    - m.x167 + m.x169 <= 0)
m.e255 = Constraint(expr= -12 * m.i31 + 12 * m.i32 + 0.5 * m.x97 + 0.5 * m.x101
    + m.x167 - m.x169 <= 12)
m.e256 = Constraint(expr= 13 * m.i31 - 13 * m.i32 + 0.5 * m.x99 + 0.5 * m.x103
    - m.x168 + m.x170 <= 13)
m.e257 = Constraint(expr= 13 * m.i31 + 13 * m.i32 + 0.5 * m.x99 + 0.5 * m.x103
    + m.x168 - m.x170 <= 26)
m.e258 = Constraint(expr= -m.x137 + m.x167 - m.x171 <= 0)
m.e259 = Constraint(expr= -m.x137 - m.x167 + m.x171 <= 0)
m.e260 = Constraint(expr= -m.x138 + m.x168 - m.x172 <= 0)
m.e261 = Constraint(expr= -m.x138 - m.x168 + m.x172 <= 0)
m.e262 = Constraint(expr= -12 * m.i33 - 12 * m.i34 + 0.5 * m.x97 + 0.5 * m.x105
    - m.x167 + m.x171 <= 0)
m.e263 = Constraint(expr= -12 * m.i33 + 12 * m.i34 + 0.5 * m.x97 + 0.5 * m.x105
    + m.x167 - m.x171 <= 12)
m.e264 = Constraint(expr= 13 * m.i33 - 13 * m.i34 + 0.5 * m.x99 + 0.5 * m.x107
    - m.x168 + m.x172 <= 13)
m.e265 = Constraint(expr= 13 * m.i33 + 13 * m.i34 + 0.5 * m.x99 + 0.5 * m.x107
    + m.x168 - m.x172 <= 26)
m.e266 = Constraint(expr= -m.x139 + m.x167 - m.x173 <= 0)
m.e267 = Constraint(expr= -m.x139 - m.x167 + m.x173 <= 0)
m.e268 = Constraint(expr= -m.x140 + m.x168 - m.x174 <= 0)
m.e269 = Constraint(expr= -m.x140 - m.x168 + m.x174 <= 0)
m.e270 = Constraint(expr= -12 * m.i35 - 12 * m.i36 + 0.5 * m.x97 + 0.5 * m.x109
    - m.x167 + m.x173 <= 0)
m.e271 = Constraint(expr= -12 * m.i35 + 12 * m.i36 + 0.5 * m.x97 + 0.5 * m.x109
    + m.x167 - m.x173 <= 12)
m.e272 = Constraint(expr= 13 * m.i35 - 13 * m.i36 + 0.5 * m.x99 + 0.5 * m.x111
    - m.x168 + m.x174 <= 13)
m.e273 = Constraint(expr= 13 * m.i35 + 13 * m.i36 + 0.5 * m.x99 + 0.5 * m.x111
    + m.x168 - m.x174 <= 26)
m.e274 = Constraint(expr= -m.x141 + m.x167 - m.x175 <= 0)
m.e275 = Constraint(expr= -m.x141 - m.x167 + m.x175 <= 0)
m.e276 = Constraint(expr= -m.x142 + m.x168 - m.x176 <= 0)
m.e277 = Constraint(expr= -m.x142 - m.x168 + m.x176 <= 0)
m.e278 = Constraint(expr= -12 * m.i37 - 12 * m.i38 + 0.5 * m.x97 + 0.5 * m.x113
    - m.x167 + m.x175 <= 0)
m.e279 = Constraint(expr= -12 * m.i37 + 12 * m.i38 + 0.5 * m.x97 + 0.5 * m.x113
    + m.x167 - m.x175 <= 12)
m.e280 = Constraint(expr= 13 * m.i37 - 13 * m.i38 + 0.5 * m.x99 + 0.5 * m.x115
    - m.x168 + m.x176 <= 13)
m.e281 = Constraint(expr= 13 * m.i37 + 13 * m.i38 + 0.5 * m.x99 + 0.5 * m.x115
    + m.x168 - m.x176 <= 26)
m.e282 = Constraint(expr= -m.x143 + m.x167 - m.x177 <= 0)
m.e283 = Constraint(expr= -m.x143 - m.x167 + m.x177 <= 0)
m.e284 = Constraint(expr= -m.x144 + m.x168 - m.x178 <= 0)
m.e285 = Constraint(expr= -m.x144 - m.x168 + m.x178 <= 0)
m.e286 = Constraint(expr= -12 * m.i39 - 12 * m.i40 + 0.5 * m.x97 + 0.5 * m.x117
    - m.x167 + m.x177 <= 0)
m.e287 = Constraint(expr= -12 * m.i39 + 12 * m.i40 + 0.5 * m.x97 + 0.5 * m.x117
    + m.x167 - m.x177 <= 12)
m.e288 = Constraint(expr= 13 * m.i39 - 13 * m.i40 + 0.5 * m.x99 + 0.5 * m.x119
    - m.x168 + m.x178 <= 13)
m.e289 = Constraint(expr= 13 * m.i39 + 13 * m.i40 + 0.5 * m.x99 + 0.5 * m.x119
    + m.x168 - m.x178 <= 26)
m.e290 = Constraint(expr= -m.x145 + m.x167 - m.x179 <= 0)
m.e291 = Constraint(expr= -m.x145 - m.x167 + m.x179 <= 0)
m.e292 = Constraint(expr= -m.x146 + m.x168 - m.x180 <= 0)
m.e293 = Constraint(expr= -m.x146 - m.x168 + m.x180 <= 0)
m.e294 = Constraint(expr= -12 * m.i41 - 12 * m.i42 + 0.5 * m.x97 + 0.5 * m.x121
    - m.x167 + m.x179 <= 0)
m.e295 = Constraint(expr= -12 * m.i41 + 12 * m.i42 + 0.5 * m.x97 + 0.5 * m.x121
    + m.x167 - m.x179 <= 12)
m.e296 = Constraint(expr= 13 * m.i41 - 13 * m.i42 + 0.5 * m.x99 + 0.5 * m.x123
    - m.x168 + m.x180 <= 13)
m.e297 = Constraint(expr= 13 * m.i41 + 13 * m.i42 + 0.5 * m.x99 + 0.5 * m.x123
    + m.x168 - m.x180 <= 26)
m.e298 = Constraint(expr= -m.x79 + m.x169 - m.x171 <= 0)
m.e299 = Constraint(expr= -m.x79 - m.x169 + m.x171 <= 0)
m.e300 = Constraint(expr= -m.x80 + m.x170 - m.x172 <= 0)
m.e301 = Constraint(expr= -m.x80 - m.x170 + m.x172 <= 0)
m.e302 = Constraint(expr= -12 * m.i43 - 12 * m.i44 + 0.5 * m.x101 + 0.5 *
    m.x105 - m.x169 + m.x171 <= 0)
m.e303 = Constraint(expr= -12 * m.i43 + 12 * m.i44 + 0.5 * m.x101 + 0.5 *
    m.x105 + m.x169 - m.x171 <= 12)
m.e304 = Constraint(expr= 13 * m.i43 - 13 * m.i44 + 0.5 * m.x103 + 0.5 * m.x107
    - m.x170 + m.x172 <= 13)
m.e305 = Constraint(expr= 13 * m.i43 + 13 * m.i44 + 0.5 * m.x103 + 0.5 * m.x107
    + m.x170 - m.x172 <= 26)
m.e306 = Constraint(expr= -m.x147 + m.x169 - m.x173 <= 0)
m.e307 = Constraint(expr= -m.x147 - m.x169 + m.x173 <= 0)
m.e308 = Constraint(expr= -m.x148 + m.x170 - m.x174 <= 0)
m.e309 = Constraint(expr= -m.x148 - m.x170 + m.x174 <= 0)
m.e310 = Constraint(expr= -12 * m.i45 - 12 * m.i46 + 0.5 * m.x101 + 0.5 *
    m.x109 - m.x169 + m.x173 <= 0)
m.e311 = Constraint(expr= -12 * m.i45 + 12 * m.i46 + 0.5 * m.x101 + 0.5 *
    m.x109 + m.x169 - m.x173 <= 12)
m.e312 = Constraint(expr= 13 * m.i45 - 13 * m.i46 + 0.5 * m.x103 + 0.5 * m.x111
    - m.x170 + m.x174 <= 13)
m.e313 = Constraint(expr= 13 * m.i45 + 13 * m.i46 + 0.5 * m.x103 + 0.5 * m.x111
    + m.x170 - m.x174 <= 26)
m.e314 = Constraint(expr= -m.x149 + m.x169 - m.x175 <= 0)
m.e315 = Constraint(expr= -m.x149 - m.x169 + m.x175 <= 0)
m.e316 = Constraint(expr= -m.x150 + m.x170 - m.x176 <= 0)
m.e317 = Constraint(expr= -m.x150 - m.x170 + m.x176 <= 0)
m.e318 = Constraint(expr= -12 * m.i47 - 12 * m.i48 + 0.5 * m.x101 + 0.5 *
    m.x113 - m.x169 + m.x175 <= 0)
m.e319 = Constraint(expr= -12 * m.i47 + 12 * m.i48 + 0.5 * m.x101 + 0.5 *
    m.x113 + m.x169 - m.x175 <= 12)
m.e320 = Constraint(expr= 13 * m.i47 - 13 * m.i48 + 0.5 * m.x103 + 0.5 * m.x115
    - m.x170 + m.x176 <= 13)
m.e321 = Constraint(expr= 13 * m.i47 + 13 * m.i48 + 0.5 * m.x103 + 0.5 * m.x115
    + m.x170 - m.x176 <= 26)
m.e322 = Constraint(expr= -m.x151 + m.x169 - m.x177 <= 0)
m.e323 = Constraint(expr= -m.x151 - m.x169 + m.x177 <= 0)
m.e324 = Constraint(expr= -m.x152 + m.x170 - m.x178 <= 0)
m.e325 = Constraint(expr= -m.x152 - m.x170 + m.x178 <= 0)
m.e326 = Constraint(expr= -12 * m.i49 - 12 * m.i50 + 0.5 * m.x101 + 0.5 *
    m.x117 - m.x169 + m.x177 <= 0)
m.e327 = Constraint(expr= -12 * m.i49 + 12 * m.i50 + 0.5 * m.x101 + 0.5 *
    m.x117 + m.x169 - m.x177 <= 12)
m.e328 = Constraint(expr= 13 * m.i49 - 13 * m.i50 + 0.5 * m.x103 + 0.5 * m.x119
    - m.x170 + m.x178 <= 13)
m.e329 = Constraint(expr= 13 * m.i49 + 13 * m.i50 + 0.5 * m.x103 + 0.5 * m.x119
    + m.x170 - m.x178 <= 26)
m.e330 = Constraint(expr= -m.x153 + m.x169 - m.x179 <= 0)
m.e331 = Constraint(expr= -m.x153 - m.x169 + m.x179 <= 0)
m.e332 = Constraint(expr= -m.x154 + m.x170 - m.x180 <= 0)
m.e333 = Constraint(expr= -m.x154 - m.x170 + m.x180 <= 0)
m.e334 = Constraint(expr= -12 * m.i51 - 12 * m.i52 + 0.5 * m.x101 + 0.5 *
    m.x121 - m.x169 + m.x179 <= 0)
m.e335 = Constraint(expr= -12 * m.i51 + 12 * m.i52 + 0.5 * m.x101 + 0.5 *
    m.x121 + m.x169 - m.x179 <= 12)
m.e336 = Constraint(expr= 13 * m.i51 - 13 * m.i52 + 0.5 * m.x103 + 0.5 * m.x123
    - m.x170 + m.x180 <= 13)
m.e337 = Constraint(expr= 13 * m.i51 + 13 * m.i52 + 0.5 * m.x103 + 0.5 * m.x123
    + m.x170 - m.x180 <= 26)
m.e338 = Constraint(expr= -m.x81 + m.x171 - m.x173 <= 0)
m.e339 = Constraint(expr= -m.x81 - m.x171 + m.x173 <= 0)
m.e340 = Constraint(expr= -m.x82 + m.x172 - m.x174 <= 0)
m.e341 = Constraint(expr= -m.x82 - m.x172 + m.x174 <= 0)
m.e342 = Constraint(expr= -12 * m.i53 - 12 * m.i54 + 0.5 * m.x105 + 0.5 *
    m.x109 - m.x171 + m.x173 <= 0)
m.e343 = Constraint(expr= -12 * m.i53 + 12 * m.i54 + 0.5 * m.x105 + 0.5 *
    m.x109 + m.x171 - m.x173 <= 12)
m.e344 = Constraint(expr= 13 * m.i53 - 13 * m.i54 + 0.5 * m.x107 + 0.5 * m.x111
    - m.x172 + m.x174 <= 13)
m.e345 = Constraint(expr= 13 * m.i53 + 13 * m.i54 + 0.5 * m.x107 + 0.5 * m.x111
    + m.x172 - m.x174 <= 26)
m.e346 = Constraint(expr= -m.x155 + m.x171 - m.x175 <= 0)
m.e347 = Constraint(expr= -m.x155 - m.x171 + m.x175 <= 0)
m.e348 = Constraint(expr= -m.x156 + m.x172 - m.x176 <= 0)
m.e349 = Constraint(expr= -m.x156 - m.x172 + m.x176 <= 0)
m.e350 = Constraint(expr= -12 * m.i55 - 12 * m.i56 + 0.5 * m.x105 + 0.5 *
    m.x113 - m.x171 + m.x175 <= 0)
m.e351 = Constraint(expr= -12 * m.i55 + 12 * m.i56 + 0.5 * m.x105 + 0.5 *
    m.x113 + m.x171 - m.x175 <= 12)
m.e352 = Constraint(expr= 13 * m.i55 - 13 * m.i56 + 0.5 * m.x107 + 0.5 * m.x115
    - m.x172 + m.x176 <= 13)
m.e353 = Constraint(expr= 13 * m.i55 + 13 * m.i56 + 0.5 * m.x107 + 0.5 * m.x115
    + m.x172 - m.x176 <= 26)
m.e354 = Constraint(expr= -m.x157 + m.x171 - m.x177 <= 0)
m.e355 = Constraint(expr= -m.x157 - m.x171 + m.x177 <= 0)
m.e356 = Constraint(expr= -m.x158 + m.x172 - m.x178 <= 0)
m.e357 = Constraint(expr= -m.x158 - m.x172 + m.x178 <= 0)
m.e358 = Constraint(expr= -12 * m.i57 - 12 * m.i58 + 0.5 * m.x105 + 0.5 *
    m.x117 - m.x171 + m.x177 <= 0)
m.e359 = Constraint(expr= -12 * m.i57 + 12 * m.i58 + 0.5 * m.x105 + 0.5 *
    m.x117 + m.x171 - m.x177 <= 12)
m.e360 = Constraint(expr= 13 * m.i57 - 13 * m.i58 + 0.5 * m.x107 + 0.5 * m.x119
    - m.x172 + m.x178 <= 13)
m.e361 = Constraint(expr= 13 * m.i57 + 13 * m.i58 + 0.5 * m.x107 + 0.5 * m.x119
    + m.x172 - m.x178 <= 26)
m.e362 = Constraint(expr= -m.x159 + m.x171 - m.x179 <= 0)
m.e363 = Constraint(expr= -m.x159 - m.x171 + m.x179 <= 0)
m.e364 = Constraint(expr= -m.x160 + m.x172 - m.x180 <= 0)
m.e365 = Constraint(expr= -m.x160 - m.x172 + m.x180 <= 0)
m.e366 = Constraint(expr= -12 * m.i59 - 12 * m.i60 + 0.5 * m.x105 + 0.5 *
    m.x121 - m.x171 + m.x179 <= 0)
m.e367 = Constraint(expr= -12 * m.i59 + 12 * m.i60 + 0.5 * m.x105 + 0.5 *
    m.x121 + m.x171 - m.x179 <= 12)
m.e368 = Constraint(expr= 13 * m.i59 - 13 * m.i60 + 0.5 * m.x107 + 0.5 * m.x123
    - m.x172 + m.x180 <= 13)
m.e369 = Constraint(expr= 13 * m.i59 + 13 * m.i60 + 0.5 * m.x107 + 0.5 * m.x123
    + m.x172 - m.x180 <= 26)
m.e370 = Constraint(expr= -m.x83 + m.x173 - m.x175 <= 0)
m.e371 = Constraint(expr= -m.x83 - m.x173 + m.x175 <= 0)
m.e372 = Constraint(expr= -m.x84 + m.x174 - m.x176 <= 0)
m.e373 = Constraint(expr= -m.x84 - m.x174 + m.x176 <= 0)
m.e374 = Constraint(expr= -12 * m.i61 - 12 * m.i62 + 0.5 * m.x109 + 0.5 *
    m.x113 - m.x173 + m.x175 <= 0)
m.e375 = Constraint(expr= -12 * m.i61 + 12 * m.i62 + 0.5 * m.x109 + 0.5 *
    m.x113 + m.x173 - m.x175 <= 12)
m.e376 = Constraint(expr= 13 * m.i61 - 13 * m.i62 + 0.5 * m.x111 + 0.5 * m.x115
    - m.x174 + m.x176 <= 13)
m.e377 = Constraint(expr= 13 * m.i61 + 13 * m.i62 + 0.5 * m.x111 + 0.5 * m.x115
    + m.x174 - m.x176 <= 26)
m.e378 = Constraint(expr= -m.x161 + m.x173 - m.x177 <= 0)
m.e379 = Constraint(expr= -m.x161 - m.x173 + m.x177 <= 0)
m.e380 = Constraint(expr= -m.x162 + m.x174 - m.x178 <= 0)
m.e381 = Constraint(expr= -m.x162 - m.x174 + m.x178 <= 0)
m.e382 = Constraint(expr= -12 * m.i63 - 12 * m.i64 + 0.5 * m.x109 + 0.5 *
    m.x117 - m.x173 + m.x177 <= 0)
m.e383 = Constraint(expr= -12 * m.i63 + 12 * m.i64 + 0.5 * m.x109 + 0.5 *
    m.x117 + m.x173 - m.x177 <= 12)
m.e384 = Constraint(expr= 13 * m.i63 - 13 * m.i64 + 0.5 * m.x111 + 0.5 * m.x119
    - m.x174 + m.x178 <= 13)
m.e385 = Constraint(expr= 13 * m.i63 + 13 * m.i64 + 0.5 * m.x111 + 0.5 * m.x119
    + m.x174 - m.x178 <= 26)
m.e386 = Constraint(expr= -m.x163 + m.x173 - m.x179 <= 0)
m.e387 = Constraint(expr= -m.x163 - m.x173 + m.x179 <= 0)
m.e388 = Constraint(expr= -m.x164 + m.x174 - m.x180 <= 0)
m.e389 = Constraint(expr= -m.x164 - m.x174 + m.x180 <= 0)
m.e390 = Constraint(expr= -12 * m.i65 - 12 * m.i66 + 0.5 * m.x109 + 0.5 *
    m.x121 - m.x173 + m.x179 <= 0)
m.e391 = Constraint(expr= -12 * m.i65 + 12 * m.i66 + 0.5 * m.x109 + 0.5 *
    m.x121 + m.x173 - m.x179 <= 12)
m.e392 = Constraint(expr= 13 * m.i65 - 13 * m.i66 + 0.5 * m.x111 + 0.5 * m.x123
    - m.x174 + m.x180 <= 13)
m.e393 = Constraint(expr= 13 * m.i65 + 13 * m.i66 + 0.5 * m.x111 + 0.5 * m.x123
    + m.x174 - m.x180 <= 26)
m.e394 = Constraint(expr= -m.x85 + m.x175 - m.x177 <= 0)
m.e395 = Constraint(expr= -m.x85 - m.x175 + m.x177 <= 0)
m.e396 = Constraint(expr= -m.x86 + m.x176 - m.x178 <= 0)
m.e397 = Constraint(expr= -m.x86 - m.x176 + m.x178 <= 0)
m.e398 = Constraint(expr= -12 * m.i67 - 12 * m.i68 + 0.5 * m.x113 + 0.5 *
    m.x117 - m.x175 + m.x177 <= 0)
m.e399 = Constraint(expr= -12 * m.i67 + 12 * m.i68 + 0.5 * m.x113 + 0.5 *
    m.x117 + m.x175 - m.x177 <= 12)
m.e400 = Constraint(expr= 13 * m.i67 - 13 * m.i68 + 0.5 * m.x115 + 0.5 * m.x119
    - m.x176 + m.x178 <= 13)
m.e401 = Constraint(expr= 13 * m.i67 + 13 * m.i68 + 0.5 * m.x115 + 0.5 * m.x119
    + m.x176 - m.x178 <= 26)
m.e402 = Constraint(expr= -m.x165 + m.x175 - m.x179 <= 0)
m.e403 = Constraint(expr= -m.x165 - m.x175 + m.x179 <= 0)
m.e404 = Constraint(expr= -m.x166 + m.x176 - m.x180 <= 0)
m.e405 = Constraint(expr= -m.x166 - m.x176 + m.x180 <= 0)
m.e406 = Constraint(expr= -12 * m.i69 - 12 * m.i70 + 0.5 * m.x113 + 0.5 *
    m.x121 - m.x175 + m.x179 <= 0)
m.e407 = Constraint(expr= -12 * m.i69 + 12 * m.i70 + 0.5 * m.x113 + 0.5 *
    m.x121 + m.x175 - m.x179 <= 12)
m.e408 = Constraint(expr= 13 * m.i69 - 13 * m.i70 + 0.5 * m.x115 + 0.5 * m.x123
    - m.x176 + m.x180 <= 13)
m.e409 = Constraint(expr= 13 * m.i69 + 13 * m.i70 + 0.5 * m.x115 + 0.5 * m.x123
    + m.x176 - m.x180 <= 26)
m.e410 = Constraint(expr= -m.x87 + m.x177 - m.x179 <= 0)
m.e411 = Constraint(expr= -m.x87 - m.x177 + m.x179 <= 0)
m.e412 = Constraint(expr= -m.x88 + m.x178 - m.x180 <= 0)
m.e413 = Constraint(expr= -m.x88 - m.x178 + m.x180 <= 0)
m.e414 = Constraint(expr= -12 * m.i71 - 12 * m.i72 + 0.5 * m.x117 + 0.5 *
    m.x121 - m.x177 + m.x179 <= 0)
m.e415 = Constraint(expr= -12 * m.i71 + 12 * m.i72 + 0.5 * m.x117 + 0.5 *
    m.x121 + m.x177 - m.x179 <= 12)
m.e416 = Constraint(expr= 13 * m.i71 - 13 * m.i72 + 0.5 * m.x119 + 0.5 * m.x123
    - m.x178 + m.x180 <= 13)
m.e417 = Constraint(expr= 13 * m.i71 + 13 * m.i72 + 0.5 * m.x119 + 0.5 * m.x123
    + m.x178 - m.x180 <= 26)
m.e418 = Constraint(expr= 16 / m.x93 - m.x95 <= 0)
m.e419 = Constraint(expr= 16 / m.x95 - m.x93 <= 0)
m.e420 = Constraint(expr= 16 / m.x94 - m.x96 <= 0)
m.e421 = Constraint(expr= 16 / m.x96 - m.x94 <= 0)
m.e422 = Constraint(expr= 16 / m.x97 - m.x99 <= 0)
m.e423 = Constraint(expr= 16 / m.x99 - m.x97 <= 0)
m.e424 = Constraint(expr= 36 / m.x101 - m.x103 <= 0)
m.e425 = Constraint(expr= 36 / m.x103 - m.x101 <= 0)
m.e426 = Constraint(expr= 36 / m.x105 - m.x107 <= 0)
m.e427 = Constraint(expr= 36 / m.x107 - m.x105 <= 0)
m.e428 = Constraint(expr= 9 / m.x109 - m.x111 <= 0)
m.e429 = Constraint(expr= 9 / m.x111 - m.x109 <= 0)
m.e430 = Constraint(expr= 9 / m.x113 - m.x115 <= 0)
m.e431 = Constraint(expr= 9 / m.x115 - m.x113 <= 0)
m.e432 = Constraint(expr= 9 / m.x117 - m.x119 <= 0)
m.e433 = Constraint(expr= 9 / m.x119 - m.x117 <= 0)
m.e434 = Constraint(expr= 9 / m.x121 - m.x123 <= 0)
m.e435 = Constraint(expr= 9 / m.x123 - m.x121 <= 0)

from pyomo import opt as po
solver = po.SolverFactory("scip")
solver.solve(m, tee=True, keepfiles=True)
