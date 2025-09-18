""" Created on Tue Oct  3 15:20:10 2023
    @author: dcupolillo """

import random


def generate_uuid(
) -> tuple:

    uuid_hex = ''.join(random.choices('0123456789ABCDEF', k=16))
    uuid_uint64 = int(uuid_hex, 16)
    uuid_str = "{:.9e}".format(uuid_uint64)

    return uuid_hex, uuid_str