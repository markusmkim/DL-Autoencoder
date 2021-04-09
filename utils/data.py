from math import floor

def split_data_sets(frac_d1, data, targets):
    split_index = int(floor(frac_d1 * len(data)))
    return (data[:split_index], targets[:split_index]), (data[split_index:], targets[split_index:])