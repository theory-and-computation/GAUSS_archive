#----------------------------#
# calc_colvar_forces_tcl.tcl #
#----------------------------#

set model_file "model_data.txt"

# extract model data from file ----------------------------#
puts "\nEXTRACTING DATA\n"
set model [open $model_file r]
set cv1_arr {}
set cv2_arr {}
set f1_arr {}
set f2_arr {}
gets $model header
while {[gets $model line] >= 0} {
    set data [split $line]
    if {[llength $data] == 4} {
        lappend cv1_arr [expr {double([lindex $data 0])}]
        lappend cv2_arr [expr {double([lindex $data 1])}]
        lappend f1_arr [expr {double([lindex $data 2])}]
        lappend f2_arr [expr {double([lindex $data 3])}]
    } else {
		puts stderr "Warning: skipping malformed line: $line"
	}
}
close $model

# get dimensions of cv-space
set uniq_cv_vals [lsort -real -unique $cv1_arr]
set min 1000
set max 0
foreach cv $uniq_cv_vals {
    if {$cv < $min} {
        set min $cv
    }
    if {$cv > $max} {
        set max $cv
    }
}
set n_bin [expr [llength $uniq_cv_vals] - 1]
set bin_width [expr ($max - $min) / $n_bin]

#----------------------------------------------------------#

proc _map_cv_to_f { cv1 cv2 } {
    global f1_arr
    global f2_arr
    global n_bin
    global bin_width

    set cv1_bin_index [expr {round( ($cv1 + 180) / $bin_width )}]
	set cv2_bin_index [expr {round( ($cv2 + 180) / $bin_width )}]
	
	set index [expr ($cv2_bin_index * ($n_bin + 1)) + $cv1_bin_index]
	
	set f1 [lindex $f1_arr $index]
	set f2 [lindex $f2_arr $index]
	
	puts "CV_1: $cv1   CV_2: $cv2   index: $index   force_1: $f1   force_2: $f2"
    
    return [list $f1 $f2]
}

#_map_cv_to_f 31 137

## search algorithm ----------------------------------------#
#proc _map_cv_to_f { cv1 cv2 } {
#	global cv1_arr
#	global f1_arr
#	global cv2_arr
#	global f2_arr
#
#	set lo 0
#	set hi [expr {[llength $cv1_arr] - 1}]
#
#	while {$lo <= $hi} {
#
#        # search cv2 first
#		if {[expr $hi - $lo] == 1} { 
#			set dlo [expr {abs($cv2 - [lindex $cv2_arr $lo])}]
#			set dhi [expr {abs($cv2 - [lindex $cv2_arr $hi])}]
#			set closer_id [expr {($dlo < $dhi) * $lo + ($dlo >= $dhi) * $hi}]
#			set f2 [lindex $f2_arr $closer_id]
#			break
#
#		} else {
#			set idx_mid [expr {int(($lo + $hi) / 2)}]
#			set cv2_mid [expr floor([lindex $cv2_arr $idx_mid] * 100) / 100]
#            
#			if {$cv2 == $cv2_mid} {
#				set f [lindex $f_arr $mid]
#				break
#			} elseif {$cv < $cv_mid} {
#				set hi $mid
#			} else {
#				set lo $mid
#			}
#		}
#	}
#	return $f1 $f2
#}
#----------------------------------------------------------#



# main procedure ------------------------------------------#
proc calc_colvar_forces { step } {
	set cv1 [expr floor([cv colvar phi value] * 100) / 100]
	set cv2 [expr floor([cv colvar psi value] * 100) / 100]

	#set f1 f2 [_map_cv_to_f $cv1 $cv2]
    lassign [_map_cv_to_f $cv1 $cv2] f1 f2

    cv colvar phi addforce $f1
    cv colvar psi addforce $f2
}
#----------------------------------------------------------#