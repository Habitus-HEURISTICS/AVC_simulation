import numpy as np

gfreq = 52
num_days = 365

def DOY_to_bin(DOYs, freq = gfreq, num_days = num_days):
    bins = np.linspace(0, num_days, freq)
    bins = np.digitize(DOYs, bins)
    return bins

# Takes an array of start and end dates and returns a vector where bins that contain the date range have 1
def range_to_prob(drange, num_rows, freq = gfreq, num_days = num_days):
    # Takes a DOY range and returns some probability bucket
    end_bins = DOY_to_bin(wraparound(drange))
    num_bins = end_bins[1] - end_bins[0]
    probs = np.zeros((num_rows, freq))

    same_year_idx = (end_bins[0][:, None] <= end_bins[1][:, None])
    diff_year_idx = np.invert(same_year_idx)

    # For if your dates are within the same year
    probs[(same_year_idx) & (np.arange(probs.shape[1]) >= end_bins[0][:, None]) & (np.arange(probs.shape[1]) <= end_bins[1][:, None])] = 1
    # For if your dates include the new year
    probs[(diff_year_idx) & (np.arange(probs.shape[1]) >= end_bins[0][:, None]) & (np.arange(probs.shape[1]) <= num_days)] = 1
    probs[(diff_year_idx) & (np.arange(probs.shape[1]) <= end_bins[1][:, None]) & (np.arange(probs.shape[1]) >= 0)] = 1
    return probs


def wraparound(drange, num_days = 365):
    if type(drange) == int:
        return drange % 365
    else:
        drange[drange > num_days] = drange[drange > num_days] - num_days
        drange[drange < 0] = drange[drange < 0] + num_days
    return drange



def get_season_bins(season, num_rows):
    if season == 'cold/dry':
        return range_to_prob(np.array([np.array([274]*num_rows), np.array([59]*num_rows)]), num_rows) # Oct to Feb
    elif season == 'hot/dry':
        return range_to_prob(np.array([np.array([60]*num_rows), np.array([181]*num_rows)]), num_rows) # Mar to Jun
    else:
        return range_to_prob(np.array([np.array([182]*num_rows), np.array([273]*num_rows)]), num_rows) # Jul to Sep


def get_harvest_bins(season, num_rows):
    if season == 'cold/dry':
        return range_to_prob(np.array([np.array([32]*num_rows), np.array([59]*num_rows)]), num_rows) # Month of Feb
    elif season == 'hot/dry':
        return range_to_prob(np.array([np.array([152]*num_rows), np.array([181]*num_rows)]), num_rows) # Month of Jun
    else:
        return range_to_prob(np.array([np.array([244]*num_rows), np.array([273]*num_rows)]), num_rows) # Month of Sep









# print()
# print("Welcome to your tiny timing function tutorial")
# print("Set gfreq to 52 for weekly bins")
# print("Using DOY_to_bin to convert the 50th day of the year to a week bin: ", DOY_to_bin(50))
# print("Using DOY_to_bin to convert the 50th, 80th and 320th days of the year to week bins: ", DOY_to_bin(np.array([50, 80, 320])))
# print()
# print("Using wraparound function to convert -5 into day of year: ", wraparound(-5))
# print("Using wraparound function to convert 390 into day of year: ", wraparound(390))
# print()
# pop = Pop(size=1) # Just test for one person
# iprobs = np.array([[0]*20 + [1] + [0]*31]) # Say your crop is mature in Week 21
# harvest = pop.make_decision(
#     name = 'harvest_timing', 
#     options = list(range(0,gfreq)),
#     init_probs = iprobs
#     )
# print("Using range_to_prob to convert a date range into a probability vector where bins that contain the date range have probability 1/num_bins.")
# print("         Date range of days to be converted to probability over weeks ([start day], [end day]): ")
# print(np.array([np.array([70]), np.array([80])]))
# print("         Probability over weeks given date range: ")
# print(range_to_prob(harvest, np.array([np.array([70]), np.array([80])])))
# print(range_to_prob(harvest, np.array([np.array([360]), np.array([10])])))
# print("You might use this as part of an Influence to nudge a probability vector closer to a range of dates (e.g. if harvesters are all equally available within some time period)")


