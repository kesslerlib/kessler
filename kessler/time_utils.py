import datetime
def getCcsdsTimeFormat(timeString):

    '''
    Adapted by Andrew Ng, 18/3/2022.
    Original MATLAB source code found at: https://github.com/nasa/CARA_Analysis_Tools/blob/master/two-dimension_Pc/Main/TransformationCode/TimeTransformations/getCcsdsTimeFormat.m
    The CCSDS time format is required to be of the general form

    yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z]
    (1) The date and time fields are separated by a "T".
    (2) The date field has a four digit year followed by either a two digit 
        month and two digit day, or a three digit day-of-year.  
    (3) The year, month, day, and day-of-year fields are separated by a dash.
    (4) The hours, minutes and seconds fields are each two digits separated 
        by colons.
    (5) The fraction of seconds is optional and can have any number of
        digits.
    (6) If a fraction of seconds is provided, it is separated from the two
        digit seconds by a period.
    (7) The time string can end with an optional "Z" time zone indicator
    '''
    timeFormat = []
    numT = timeString.count('T')
    if numT == -1:
        # Case when this is 'T' does not exist in timeString
        print(f"*** Error -- Invalid CCSDS time string: {timeString}")
        print(f"    No 'T' separator found between date and time portions of the string\n")
        return
    elif numT > 1:
        print(f"*** Error -- Invalid CCSDS time string: {timeString} \n")
        print(f"    More than one 'T' separator found between date and time portions of the string\n")
        return
    idxT = timeString.find('T')
    if idxT ==10:
        timeFormat = "yyyy-mm-ddTHH:MM:SS"
    elif idxT ==8:
        timeFormat = "yyyy-DDDTHH:MM:SS"
    else: 
        print(f"*** Error -- Invalid CCSDS time string: {timeString} \n", timeString)
        print(f"    Date format not one of yyyy-mm-dd or yyyy-DDD\n")
        return
    # % Check if 'Z' time zone indicator appended to the string
    if timeString[-1]=='Z':
        zOpt = True
    else:
        zOpt = False
    # % Find location of the fraction of seconds decimal separator
    numDecimal = timeString.count('.')
    if numDecimal > 1:
        print(f"*** Error -- Invalid CCSDS time string: {timeString} \n")
        print(f"    More than one fraction of seconds decimal separator ('.') found.\n")
        timeFormat = []
        return
    idxDecimal = timeString.find('.')
    nfrac = 0
    if numDecimal != 0:
        if zOpt:
            nfrac = len(timeString) - 1 - idxDecimal -1
        else: 
            nfrac = len(timeString) - 1 - idxDecimal
    if nfrac > 0:
        fracStr = '.' + ('F'*nfrac)
    else:
        fracStr = ""
    if zOpt:
        fracStr = fracStr+'Z'
    timeFormat = timeFormat+fracStr
    return timeFormat

def DOY2Date(DOY, YEAR):
    '''
    Written by Andrew Ng, 18/03/2022, 
    Based on source code @ https://github.com/nasa/CARA_Analysis_Tools/blob/master/two-dimension_Pc/Main/TransformationCode/TimeTransformations/DOY2Date.m
    Use the datetime python package. 
    DOY2DATE  - Converts Day of Year (DOY) to date number and full 
            calendar date. 

    '''
    # Calculate datetime format
    DateNum = datetime.datetime(int(YEAR), 1, 1) + datetime.timedelta(int(DOY) - 1)
    # Split datetime object into a date list
    DateVec = [DateNum.year, DateNum.month, DateNum.day, DateNum.hour, DateNum.minute]
    return DateNum, DateVec