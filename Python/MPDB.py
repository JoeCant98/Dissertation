#!/usr/bin/env python



from pymatgen import MPRester



apikey = 'NthZ9vFMzvVhRxGnFG3L'

mpr = MPRester(apikey)


formulas = []

mpid_list = []

data = mpr.query(criteria={"pretty_formula": "?1?1O3"},properties=["pretty_formula", "icsd_ids","e_above_hull","energy_per_atom","diel","piezo","is_compatible""material_id","elements"])




for mat in data:

    # print(mat)

    if (mat["is_compatible"]):


            formulas.append(mat["pretty_formula"])

            mpid_list.append(mat["material_id"])

            print(mat["elements"])

for formula in formulas:

    print(formula)



print(mpid_list)

print(len(mpid_list))

print(len(set(mpid_list)))