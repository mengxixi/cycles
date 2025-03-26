import numpy as np


def bound(method, L, beta):
    if method == "HB":
        return 2 * (1 + beta) / L
    elif method == "NAG":
        return (1 + 1 / (1 + 2 * beta)) / L
    elif method == "GD":
        return 2 / L
    elif method == "TOS":
        return 2 / L
    else:
        raise Exception
    

def merge_overlapping_intervals(arr):
    n = len(arr)

    arr.sort()
    res = []

    # Checking for all possible overlaps
    for i in range(n):
        start = arr[i][0]
        end = arr[i][1]

        # Skipping already merged intervals
        if res and res[-1][1] >= end:
            continue

        # Find the end of the merged range
        for j in range(i + 1, n):
            if arr[j][0] <= end:
                end = max(end, arr[j][1])
                
        res.append([start, end])
    
    return res


def complement_of_intervals_within_bounds(intervals, b_min, b_max):
    if len(intervals) == 0:
        return [(b_min, b_max)]
    
    res = []

    for i in range(len(intervals)):
        l = intervals[i][0]
        if i == 0 and l > b_min:
            res += [ (b_min, l) ]
            continue
        
        r = intervals[i-1][1]
        res += [ (r, l) ]
        
    r = intervals[-1][1]
    if r < b_max:
        res += [ (r, b_max) ]

    return res


def is_valid(gamma, beta, L):
    return gamma*L < 2 * (1+beta) and gamma*L >= 0


def valid_for_another_K(beta, gamma, K, mu, kappa):
    # return False

    theta = 2*np.pi/K
    cos = np.cos(theta)
    
    a = mu**2
    b = - 2*mu * (beta - cos + kappa * (1-beta*cos))
    c = 2*kappa*(1-cos)*(1+beta**2-2*beta*cos)
    
    return a * gamma**2 + b * gamma + c <= 0


def get_cycle_intervals_for_beta(mu, L, beta, K_max=10):
    intervals = []
    
    kappa = mu/L
    omega = 1 # change to 2 to get multiples of fractions
    K_max = omega*10
    start = 3
    K_range = [i/omega for i in range(omega*start, K_max+1)]
    
    for K in K_range:
        theta = 2*np.pi/K
        cos = np.cos(theta)
        a = mu**2
        b = - 2*mu * (beta - cos + kappa * (1-beta*cos))
        c = 2*kappa*(1-cos)*(1+beta**2-2*beta*cos)
        
        if b**2 < 4*a*c:
            continue
        
        gamma1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        gamma2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        
        v1 = is_valid(gamma1, beta, L)
        v2 = is_valid(gamma2, beta, L)
        
        if not (v1 or v2):
            # neither are valid, move on to the next K
            continue
        
        # clip them to be between bounds
        bmax = bound("HB", L, beta)
        gamma1 = np.clip(gamma1, 0, bmax)
        gamma2 = np.clip(gamma2, 0, bmax)
        
        intervals += [ (gamma1, gamma2) ]
            
    # now merge intervals if they overlap
    merged_intervals = merge_overlapping_intervals(intervals)
    
    return merged_intervals
    

def get_cycle_tunings(mu, L, betas):
    kappa = mu/L
    omega = 1 # change to 2 to get multiples of fractions
    K_max = omega*10
    start = 3
    K_range = [i/omega for i in range(omega*start, K_max+1)]
    valid_tunings = {}
    
    for i, K in enumerate(K_range):

        valid_betas = []
        valid_Lgammas = []

        theta = 2*np.pi/K
        cos = np.cos(theta)
        
        for beta in betas:
            a = mu**2
            b = - 2*mu * (beta - cos + kappa * (1-beta*cos))
            c = 2*kappa*(1-cos)*(1+beta**2-2*beta*cos)
            
            if b**2 < 4*a*c:
                continue
            
            gamma1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            gamma2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            
            v1 = is_valid(gamma1, beta, L)
            v2 = is_valid(gamma2, beta, L)
            
            # TODO: instead of checking if this gamma is already included
            #       just use the overlapping intervals approach above
                
            if v1:
                included = False
                for k in K_range[:i]:
                    included = included or valid_for_another_K(beta, gamma1, k, mu, kappa)
                
                if not included:
                    valid_betas.append(beta)
                    valid_Lgammas.append(gamma1*L)
            
            if v2:
                included = False
                for k in K_range[:i]:
                    included = included or valid_for_another_K(beta, gamma2, k, mu, kappa)
                
                if not included:
                    valid_betas.append(beta)
                    valid_Lgammas.append(gamma2*L)

        valid_tunings[str(K)] = (K, valid_betas, valid_Lgammas)
        
    return valid_tunings

