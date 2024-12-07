import pandas as pd

def read_and_label_fragments(file_path):
    """
    Read the first 4 columns and 'sigma' column of the file (assuming space or tab-separated).
    Returns a DataFrame with columns: Fragment Z, Fragment A, Projectile Z, Projectile A, sigma.
    """
    try:
        # Load the first 4 columns and the 100th column (assuming it's 'sigma')
        data = pd.read_csv(file_path, delim_whitespace=True, usecols=[0, 1, 2, 3, 100],
                           header=None, names=["Fragment Z", "Fragment A", "Projectile Z", "Projectile A", "sigma"])
        return data
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def find_matching_row(data, frag_z, frag_a, proj_z, proj_a):
    """
    Find the row where 'Fragment Z', 'Fragment A', 'Projectile Z', and 'Projectile A' match the input values.
    Returns the matching rows as a DataFrame, or None if no match is found.
    """
    try:
        # Filter the DataFrame to find the matching row(s)
        matching_rows = data[(data['Fragment Z'] == frag_z) &
                             (data['Fragment A'] == frag_a) &
                             (data['Projectile Z'] == proj_z) &
                             (data['Projectile A'] == proj_a)]
        
        if not matching_rows.empty:
            return matching_rows
        else:
            print(f"No matching row found for Fragment Z={frag_z}, Fragment A={frag_a}, "
                  f"Projectile Z={proj_z}, Projectile A={proj_a}.")
            return None
    except Exception as e:
        print(f"Error occurred during row search: {e}")
        return None

def read_sigma(data, proj_z, proj_a, frag_z, frag_a):
    """
    Find the sigma value for the given 'Fragment Z', 'Fragment A', 'Projectile Z', and 'Projectile A'.
    Returns the sigma value or raises an exception if no matching row is found.
    """
    result = find_matching_row(data, frag_z, frag_a, proj_z, proj_a)
    if result is not None:
        try:
            return result["sigma"].values[0]
        except IndexError:
            print(f"No sigma value found for Fragment Z={frag_z}, Fragment A={frag_a}, "
                  f"Projectile Z={proj_z}, Projectile A={proj_a}.")
            return 0  # Return 0 if no sigma value found
    return 0

def sum_sigma(data, projectile, fragments):
    """
    Calculate the sum of sigma values for a specific projectile and a list of fragment pairs.
    Each fragment pair is a tuple of (frag_z, frag_a).
    """
    total_sigma = sum(read_sigma(data, projectile[0], projectile[1], frag[0], frag[1]) for frag in fragments)
    return total_sigma

if __name__ == "__main__":
    file_path = 'crxsecs_fragmentation_Evoli2019_cumulative.txt'
    #file_path = 'sigProdWebber03+Coste12.txt'
    data = read_and_label_fragments(file_path)

    if data is not None:
        # Calculate sigma values for different fragmentations
        sigma_o_c = sum_sigma(data, (8, 16), [(6, 12), (6, 13), (6, 14)])
        print(f'O -> C: {sigma_o_c:3.0f}')

        sigma_o_b = sum_sigma(data, (8, 16), [(5, 10), (5, 11)])
        print(f'O -> B: {sigma_o_b:3.0f}')

        sigma_c_b = sum_sigma(data, (6, 12), [(5, 10), (5, 11)])
        print(f'C -> B: {sigma_c_b:3.0f}')

        sigma_o_be10 = read_sigma(data, 8, 16, 4, 10)
        print(f'O -> Be10: {sigma_o_be10:3.0f}')

        sigma_o_be9 = read_sigma(data, 8, 16, 4, 9)
        print(f'O -> Be9: {sigma_o_be9:3.0f}')

        sigma_c_be10 = read_sigma(data, 6, 12, 4, 10)
        print(f'C -> Be10: {sigma_c_be10:3.0f}')

        sigma_c_be9 = read_sigma(data, 6, 12, 4, 9)
        print(f'C -> Be9: {sigma_c_be9:3.0f}')
