# MINLP written by GAMS Convert at 02/17/22 17:23:25
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        72        0        0       72        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#       168        0       12      156        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#       768      480      288
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.b1 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b2 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b3 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b4 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b5 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b6 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b7 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b8 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b9 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b10 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b11 = Var(within=Binary, bounds=(0,1), initialize=0)
m.b12 = Var(within=Binary, bounds=(0,1), initialize=0)
m.i13 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i14 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i15 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i16 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i17 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i18 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i19 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i20 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i21 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i22 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i23 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i24 = Var(within=Integers, bounds=(0,48), initialize=1)
m.i25 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i26 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i27 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i28 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i29 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i30 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i31 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i32 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i33 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i34 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i35 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i36 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i37 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i38 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i39 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i40 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i41 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i42 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i43 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i44 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i45 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i46 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i47 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i48 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i49 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i50 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i51 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i52 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i53 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i54 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i55 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i56 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i57 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i58 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i59 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i60 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i61 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i62 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i63 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i64 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i65 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i66 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i67 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i68 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i69 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i70 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i71 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i72 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i73 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i74 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i75 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i76 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i77 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i78 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i79 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i80 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i81 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i82 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i83 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i84 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i85 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i86 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i87 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i88 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i89 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i90 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i91 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i92 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i93 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i94 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i95 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i96 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i97 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i98 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i99 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i100 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i101 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i102 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i103 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i104 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i105 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i106 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i107 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i108 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i109 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i110 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i111 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i112 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i113 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i114 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i115 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i116 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i117 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i118 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i119 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i120 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i121 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i122 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i123 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i124 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i125 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i126 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i127 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i128 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i129 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i130 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i131 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i132 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i133 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i134 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i135 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i136 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i137 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i138 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i139 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i140 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i141 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i142 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i143 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i144 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i145 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i146 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i147 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i148 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i149 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i150 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i151 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i152 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i153 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i154 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i155 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i156 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i157 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i158 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i159 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i160 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i161 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i162 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i163 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i164 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i165 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i166 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i167 = Var(within=Integers, bounds=(0,5), initialize=1)
m.i168 = Var(within=Integers, bounds=(0,5), initialize=1)

m.obj = Objective(sense=minimize, expr= 0.1 * m.b1 + 0.2 * m.b2 + 0.3 * m.b3 +
    0.4 * m.b4 + 0.5 * m.b5 + 0.6 * m.b6 + 0.7 * m.b7 + 0.8 * m.b8 + 0.9 * m.b9
    + m.b10 + 1.1 * m.b11 + 1.2 * m.b12 + m.i13 + m.i14 + m.i15 + m.i16 +
    m.i17 + m.i18 + m.i19 + m.i20 + m.i21 + m.i22 + m.i23 + m.i24)

m.e1 = Constraint(expr= 350 * m.i25 + 450 * m.i37 + 550 * m.i49 + 650 * m.i61
    + 700 * m.i73 + 740 * m.i85 + 800 * m.i97 + 840 * m.i109 + 910 * m.i121 +
    960 * m.i133 + 1010 * m.i145 + 1060 * m.i157 <= 2100)
m.e2 = Constraint(expr= 350 * m.i26 + 450 * m.i38 + 550 * m.i50 + 650 * m.i62
    + 700 * m.i74 + 740 * m.i86 + 800 * m.i98 + 840 * m.i110 + 910 * m.i122 +
    960 * m.i134 + 1010 * m.i146 + 1060 * m.i158 <= 2100)
m.e3 = Constraint(expr= 350 * m.i27 + 450 * m.i39 + 550 * m.i51 + 650 * m.i63
    + 700 * m.i75 + 740 * m.i87 + 800 * m.i99 + 840 * m.i111 + 910 * m.i123 +
    960 * m.i135 + 1010 * m.i147 + 1060 * m.i159 <= 2100)
m.e4 = Constraint(expr= 350 * m.i28 + 450 * m.i40 + 550 * m.i52 + 650 * m.i64
    + 700 * m.i76 + 740 * m.i88 + 800 * m.i100 + 840 * m.i112 + 910 * m.i124
    + 960 * m.i136 + 1010 * m.i148 + 1060 * m.i160 <= 2100)
m.e5 = Constraint(expr= 350 * m.i29 + 450 * m.i41 + 550 * m.i53 + 650 * m.i65
    + 700 * m.i77 + 740 * m.i89 + 800 * m.i101 + 840 * m.i113 + 910 * m.i125
    + 960 * m.i137 + 1010 * m.i149 + 1060 * m.i161 <= 2100)
m.e6 = Constraint(expr= 350 * m.i30 + 450 * m.i42 + 550 * m.i54 + 650 * m.i66
    + 700 * m.i78 + 740 * m.i90 + 800 * m.i102 + 840 * m.i114 + 910 * m.i126
    + 960 * m.i138 + 1010 * m.i150 + 1060 * m.i162 <= 2100)
m.e7 = Constraint(expr= 350 * m.i31 + 450 * m.i43 + 550 * m.i55 + 650 * m.i67
    + 700 * m.i79 + 740 * m.i91 + 800 * m.i103 + 840 * m.i115 + 910 * m.i127
    + 960 * m.i139 + 1010 * m.i151 + 1060 * m.i163 <= 2100)
m.e8 = Constraint(expr= 350 * m.i32 + 450 * m.i44 + 550 * m.i56 + 650 * m.i68
    + 700 * m.i80 + 740 * m.i92 + 800 * m.i104 + 840 * m.i116 + 910 * m.i128
    + 960 * m.i140 + 1010 * m.i152 + 1060 * m.i164 <= 2100)
m.e9 = Constraint(expr= 350 * m.i33 + 450 * m.i45 + 550 * m.i57 + 650 * m.i69
    + 700 * m.i81 + 740 * m.i93 + 800 * m.i105 + 840 * m.i117 + 910 * m.i129
    + 960 * m.i141 + 1010 * m.i153 + 1060 * m.i165 <= 2100)
m.e10 = Constraint(expr= 350 * m.i34 + 450 * m.i46 + 550 * m.i58 + 650 * m.i70
    + 700 * m.i82 + 740 * m.i94 + 800 * m.i106 + 840 * m.i118 + 910 * m.i130
    + 960 * m.i142 + 1010 * m.i154 + 1060 * m.i166 <= 2100)
m.e11 = Constraint(expr= 350 * m.i35 + 450 * m.i47 + 550 * m.i59 + 650 * m.i71
    + 700 * m.i83 + 740 * m.i95 + 800 * m.i107 + 840 * m.i119 + 910 * m.i131
    + 960 * m.i143 + 1010 * m.i155 + 1060 * m.i167 <= 2100)
m.e12 = Constraint(expr= 350 * m.i36 + 450 * m.i48 + 550 * m.i60 + 650 * m.i72
    + 700 * m.i84 + 740 * m.i96 + 800 * m.i108 + 840 * m.i120 + 910 * m.i132
    + 960 * m.i144 + 1010 * m.i156 + 1060 * m.i168 <= 2100)
m.e13 = Constraint(expr= -350 * m.i25 - 450 * m.i37 - 550 * m.i49 - 650 * m.i61
    - 700 * m.i73 - 740 * m.i85 - 800 * m.i97 - 840 * m.i109 - 910 * m.i121 -
    960 * m.i133 - 1010 * m.i145 - 1060 * m.i157 <= -2000)
m.e14 = Constraint(expr= -350 * m.i26 - 450 * m.i38 - 550 * m.i50 - 650 * m.i62
    - 700 * m.i74 - 740 * m.i86 - 800 * m.i98 - 840 * m.i110 - 910 * m.i122 -
    960 * m.i134 - 1010 * m.i146 - 1060 * m.i158 <= -2000)
m.e15 = Constraint(expr= -350 * m.i27 - 450 * m.i39 - 550 * m.i51 - 650 * m.i63
    - 700 * m.i75 - 740 * m.i87 - 800 * m.i99 - 840 * m.i111 - 910 * m.i123 -
    960 * m.i135 - 1010 * m.i147 - 1060 * m.i159 <= -2000)
m.e16 = Constraint(expr= -350 * m.i28 - 450 * m.i40 - 550 * m.i52 - 650 * m.i64
    - 700 * m.i76 - 740 * m.i88 - 800 * m.i100 - 840 * m.i112 - 910 * m.i124
    - 960 * m.i136 - 1010 * m.i148 - 1060 * m.i160 <= -2000)
m.e17 = Constraint(expr= -350 * m.i29 - 450 * m.i41 - 550 * m.i53 - 650 * m.i65
    - 700 * m.i77 - 740 * m.i89 - 800 * m.i101 - 840 * m.i113 - 910 * m.i125
    - 960 * m.i137 - 1010 * m.i149 - 1060 * m.i161 <= -2000)
m.e18 = Constraint(expr= -350 * m.i30 - 450 * m.i42 - 550 * m.i54 - 650 * m.i66
    - 700 * m.i78 - 740 * m.i90 - 800 * m.i102 - 840 * m.i114 - 910 * m.i126
    - 960 * m.i138 - 1010 * m.i150 - 1060 * m.i162 <= -2000)
m.e19 = Constraint(expr= -350 * m.i31 - 450 * m.i43 - 550 * m.i55 - 650 * m.i67
    - 700 * m.i79 - 740 * m.i91 - 800 * m.i103 - 840 * m.i115 - 910 * m.i127
    - 960 * m.i139 - 1010 * m.i151 - 1060 * m.i163 <= -2000)
m.e20 = Constraint(expr= -350 * m.i32 - 450 * m.i44 - 550 * m.i56 - 650 * m.i68
    - 700 * m.i80 - 740 * m.i92 - 800 * m.i104 - 840 * m.i116 - 910 * m.i128
    - 960 * m.i140 - 1010 * m.i152 - 1060 * m.i164 <= -2000)
m.e21 = Constraint(expr= -350 * m.i33 - 450 * m.i45 - 550 * m.i57 - 650 * m.i69
    - 700 * m.i81 - 740 * m.i93 - 800 * m.i105 - 840 * m.i117 - 910 * m.i129
    - 960 * m.i141 - 1010 * m.i153 - 1060 * m.i165 <= -2000)
m.e22 = Constraint(expr= -350 * m.i34 - 450 * m.i46 - 550 * m.i58 - 650 * m.i70
    - 700 * m.i82 - 740 * m.i94 - 800 * m.i106 - 840 * m.i118 - 910 * m.i130
    - 960 * m.i142 - 1010 * m.i154 - 1060 * m.i166 <= -2000)
m.e23 = Constraint(expr= -350 * m.i35 - 450 * m.i47 - 550 * m.i59 - 650 * m.i71
    - 700 * m.i83 - 740 * m.i95 - 800 * m.i107 - 840 * m.i119 - 910 * m.i131
    - 960 * m.i143 - 1010 * m.i155 - 1060 * m.i167 <= -2000)
m.e24 = Constraint(expr= -350 * m.i36 - 450 * m.i48 - 550 * m.i60 - 650 * m.i72
    - 700 * m.i84 - 740 * m.i96 - 800 * m.i108 - 840 * m.i120 - 910 * m.i132
    - 960 * m.i144 - 1010 * m.i156 - 1060 * m.i168 <= -2000)
m.e25 = Constraint(expr= m.i25 + m.i37 + m.i49 + m.i61 + m.i73 + m.i85 + m.i97
    + m.i109 + m.i121 + m.i133 + m.i145 + m.i157 <= 5)
m.e26 = Constraint(expr= m.i26 + m.i38 + m.i50 + m.i62 + m.i74 + m.i86 + m.i98
    + m.i110 + m.i122 + m.i134 + m.i146 + m.i158 <= 5)
m.e27 = Constraint(expr= m.i27 + m.i39 + m.i51 + m.i63 + m.i75 + m.i87 + m.i99
    + m.i111 + m.i123 + m.i135 + m.i147 + m.i159 <= 5)
m.e28 = Constraint(expr= m.i28 + m.i40 + m.i52 + m.i64 + m.i76 + m.i88 + m.i100
    + m.i112 + m.i124 + m.i136 + m.i148 + m.i160 <= 5)
m.e29 = Constraint(expr= m.i29 + m.i41 + m.i53 + m.i65 + m.i77 + m.i89 + m.i101
    + m.i113 + m.i125 + m.i137 + m.i149 + m.i161 <= 5)
m.e30 = Constraint(expr= m.i30 + m.i42 + m.i54 + m.i66 + m.i78 + m.i90 + m.i102
    + m.i114 + m.i126 + m.i138 + m.i150 + m.i162 <= 5)
m.e31 = Constraint(expr= m.i31 + m.i43 + m.i55 + m.i67 + m.i79 + m.i91 + m.i103
    + m.i115 + m.i127 + m.i139 + m.i151 + m.i163 <= 5)
m.e32 = Constraint(expr= m.i32 + m.i44 + m.i56 + m.i68 + m.i80 + m.i92 + m.i104
    + m.i116 + m.i128 + m.i140 + m.i152 + m.i164 <= 5)
m.e33 = Constraint(expr= m.i33 + m.i45 + m.i57 + m.i69 + m.i81 + m.i93 + m.i105
    + m.i117 + m.i129 + m.i141 + m.i153 + m.i165 <= 5)
m.e34 = Constraint(expr= m.i34 + m.i46 + m.i58 + m.i70 + m.i82 + m.i94 + m.i106
    + m.i118 + m.i130 + m.i142 + m.i154 + m.i166 <= 5)
m.e35 = Constraint(expr= m.i35 + m.i47 + m.i59 + m.i71 + m.i83 + m.i95 + m.i107
    + m.i119 + m.i131 + m.i143 + m.i155 + m.i167 <= 5)
m.e36 = Constraint(expr= m.i36 + m.i48 + m.i60 + m.i72 + m.i84 + m.i96 + m.i108
    + m.i120 + m.i132 + m.i144 + m.i156 + m.i168 <= 5)
m.e37 = Constraint(expr= m.b1 - m.i13 <= 0)
m.e38 = Constraint(expr= m.b2 - m.i14 <= 0)
m.e39 = Constraint(expr= m.b3 - m.i15 <= 0)
m.e40 = Constraint(expr= m.b4 - m.i16 <= 0)
m.e41 = Constraint(expr= m.b5 - m.i17 <= 0)
m.e42 = Constraint(expr= m.b6 - m.i18 <= 0)
m.e43 = Constraint(expr= m.b7 - m.i19 <= 0)
m.e44 = Constraint(expr= m.b8 - m.i20 <= 0)
m.e45 = Constraint(expr= m.b9 - m.i21 <= 0)
m.e46 = Constraint(expr= m.b10 - m.i22 <= 0)
m.e47 = Constraint(expr= m.b11 - m.i23 <= 0)
m.e48 = Constraint(expr= m.b12 - m.i24 <= 0)
m.e49 = Constraint(expr= -48 * m.b1 + m.i13 <= 0)
m.e50 = Constraint(expr= -48 * m.b2 + m.i14 <= 0)
m.e51 = Constraint(expr= -48 * m.b3 + m.i15 <= 0)
m.e52 = Constraint(expr= -48 * m.b4 + m.i16 <= 0)
m.e53 = Constraint(expr= -48 * m.b5 + m.i17 <= 0)
m.e54 = Constraint(expr= -48 * m.b6 + m.i18 <= 0)
m.e55 = Constraint(expr= -48 * m.b7 + m.i19 <= 0)
m.e56 = Constraint(expr= -48 * m.b8 + m.i20 <= 0)
m.e57 = Constraint(expr= -48 * m.b9 + m.i21 <= 0)
m.e58 = Constraint(expr= -48 * m.b10 + m.i22 <= 0)
m.e59 = Constraint(expr= -48 * m.b11 + m.i23 <= 0)
m.e60 = Constraint(expr= -48 * m.b12 + m.i24 <= 0)
m.e61 = Constraint(expr= -m.i13 * m.i25 - m.i14 * m.i26 - m.i15 * m.i27 - m.i16
    * m.i28 - m.i17 * m.i29 - m.i18 * m.i30 - m.i19 * m.i31 - m.i20 * m.i32 -
    m.i21 * m.i33 - m.i22 * m.i34 - m.i23 * m.i35 - m.i24 * m.i36 <= -10)
m.e62 = Constraint(expr= -m.i13 * m.i37 - m.i14 * m.i38 - m.i15 * m.i39 - m.i16
    * m.i40 - m.i17 * m.i41 - m.i18 * m.i42 - m.i19 * m.i43 - m.i20 * m.i44 -
    m.i21 * m.i45 - m.i22 * m.i46 - m.i23 * m.i47 - m.i24 * m.i48 <= -28)
m.e63 = Constraint(expr= -m.i13 * m.i49 - m.i14 * m.i50 - m.i15 * m.i51 - m.i16
    * m.i52 - m.i17 * m.i53 - m.i18 * m.i54 - m.i19 * m.i55 - m.i20 * m.i56 -
    m.i21 * m.i57 - m.i22 * m.i58 - m.i23 * m.i59 - m.i24 * m.i60 <= -48)
m.e64 = Constraint(expr= -m.i13 * m.i61 - m.i14 * m.i62 - m.i15 * m.i63 - m.i16
    * m.i64 - m.i17 * m.i65 - m.i18 * m.i66 - m.i19 * m.i67 - m.i20 * m.i68 -
    m.i21 * m.i69 - m.i22 * m.i70 - m.i23 * m.i71 - m.i24 * m.i72 <= -28)
m.e65 = Constraint(expr= -m.i13 * m.i73 - m.i14 * m.i74 - m.i15 * m.i75 - m.i16
    * m.i76 - m.i17 * m.i77 - m.i18 * m.i78 - m.i19 * m.i79 - m.i20 * m.i80 -
    m.i21 * m.i81 - m.i22 * m.i82 - m.i23 * m.i83 - m.i24 * m.i84 <= -40)
m.e66 = Constraint(expr= -m.i13 * m.i85 - m.i14 * m.i86 - m.i15 * m.i87 - m.i16
    * m.i88 - m.i17 * m.i89 - m.i18 * m.i90 - m.i19 * m.i91 - m.i20 * m.i92 -
    m.i21 * m.i93 - m.i22 * m.i94 - m.i23 * m.i95 - m.i24 * m.i96 <= -30)
m.e67 = Constraint(expr= -m.i13 * m.i97 - m.i14 * m.i98 - m.i15 * m.i99 - m.i16
    * m.i100 - m.i17 * m.i101 - m.i18 * m.i102 - m.i19 * m.i103 - m.i20 *
    m.i104 - m.i21 * m.i105 - m.i22 * m.i106 - m.i23 * m.i107 - m.i24 * m.i108
    <= -21)
m.e68 = Constraint(expr= -m.i13 * m.i109 - m.i14 * m.i110 - m.i15 * m.i111 -
    m.i16 * m.i112 - m.i17 * m.i113 - m.i18 * m.i114 - m.i19 * m.i115 - m.i20
    * m.i116 - m.i21 * m.i117 - m.i22 * m.i118 - m.i23 * m.i119 - m.i24 *
    m.i120 <= -22)
m.e69 = Constraint(expr= -m.i13 * m.i121 - m.i14 * m.i122 - m.i15 * m.i123 -
    m.i16 * m.i124 - m.i17 * m.i125 - m.i18 * m.i126 - m.i19 * m.i127 - m.i20
    * m.i128 - m.i21 * m.i129 - m.i22 * m.i130 - m.i23 * m.i131 - m.i24 *
    m.i132 <= -8)
m.e70 = Constraint(expr= -m.i13 * m.i133 - m.i14 * m.i134 - m.i15 * m.i135 -
    m.i16 * m.i136 - m.i17 * m.i137 - m.i18 * m.i138 - m.i19 * m.i139 - m.i20
    * m.i140 - m.i21 * m.i141 - m.i22 * m.i142 - m.i23 * m.i143 - m.i24 *
    m.i144 <= -8)
m.e71 = Constraint(expr= -m.i13 * m.i145 - m.i14 * m.i146 - m.i15 * m.i147 -
    m.i16 * m.i148 - m.i17 * m.i149 - m.i18 * m.i150 - m.i19 * m.i151 - m.i20
    * m.i152 - m.i21 * m.i153 - m.i22 * m.i154 - m.i23 * m.i155 - m.i24 *
    m.i156 <= -9)
m.e72 = Constraint(expr= -m.i13 * m.i157 - m.i14 * m.i158 - m.i15 * m.i159 -
    m.i16 * m.i160 - m.i17 * m.i161 - m.i18 * m.i162 - m.i19 * m.i163 - m.i20
    * m.i164 - m.i21 * m.i165 - m.i22 * m.i166 - m.i23 * m.i167 - m.i24 *
    m.i168 <= -8)


from pyomo import opt as po
solver = po.SolverFactory("scip")
solver.options['numerics/feastol'] = 1e-6
result = solver.solve(m, tee=True)
print(result)
