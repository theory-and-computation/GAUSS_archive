#-------------#
# get_vel.tcl #
#-------------#

set root_path [lindex $argv 0]
set job_id [lindex $argv 1]
set out_path [lindex $argv 2]

# load topology and trajectory files -----------------------------------------#
mol new $root_path/components/topo/alad-wats_cross-term.psf type psf waitfor all
mol addfile $out_path/output/$job_id/$job_id.dcd type dcd waitfor all
set molcoords [molinfo top]
mol new $root_path/components/topo/alad-wats_cross-term.psf type psf waitfor all
mol addfile $out_path/output/$job_id/$job_id.veldcd type dcd waitfor all
set molvel [molinfo top]

set nfcoords [molinfo $molcoords get numframes]
set nfvel    [molinfo $molvel    get numframes]
if {$nfcoords != $nfvel} {
    error "Frame mismatch: coords=$nfcoords, veldcd=$nfvel"
}

for {set f 0} {$f < $nfcoords} {incr f} {
    set selV [atomselect $molvel "all" frame $f]
    set selC [atomselect $molcoords "all" frame $f]

    set vvals [$selV get {x y z}]
    $selC set {vx vy vz} $vvals

    $selV delete
    $selC delete
}
mol top $molcoords
#-----------------------------------------------------------------------------#


proc write {atoms_sel out_file} {
	set output [open $out_file w]
	set id_1 [format "%03d" [lindex [$atoms_sel get index] 0]]
	set id_2 [format "%03d" [lindex [$atoms_sel get index] 1]]
	set id_3 [format "%03d" [lindex [$atoms_sel get index] 2]]
	set id_4 [format "%03d" [lindex [$atoms_sel get index] 3]]
	puts $output [format " %13s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s" \
						 "{frame}" \
						 " {atom_${id_1}_vx}" " {atom_${id_1}_vy}" " {atom_${id_1}_vz}" \
						 " {atom_${id_2}_vx}" " {atom_${id_2}_vy}" " {atom_${id_2}_vz}" \
						 " {atom_${id_3}_vx}" " {atom_${id_3}_vy}" " {atom_${id_3}_vz}" \
						 " {atom_${id_4}_vx}" " {atom_${id_4}_vy}" " {atom_${id_4}_vz}"
				 ]
	set n_frames [molinfo top get numframes]
	for {set fr 0} {$fr < $n_frames} {incr fr} {
		$atoms_sel frame $fr
		set vels [$atoms_sel get {vx vy vz}]
		puts $output [format "%13d %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f" \
						$fr \
						[lindex [lindex $vels 0] 0] [lindex [lindex $vels 0] 1] [lindex [lindex $vels 0] 2] \
						[lindex [lindex $vels 1] 0] [lindex [lindex $vels 1] 1] [lindex [lindex $vels 1] 2] \
						[lindex [lindex $vels 2] 0] [lindex [lindex $vels 2] 1] [lindex [lindex $vels 2] 2] \
						[lindex [lindex $vels 3] 0] [lindex [lindex $vels 3] 1] [lindex [lindex $vels 3] 2]
					 ]
	}
	close $output
}


# main -----------------------------------#
set cv1 [atomselect top "index 4 6 8 14"]
set cv2 [atomselect top "index 6 8 14 16"]

write $cv1 $out_path/output/$job_id/$job_id.cv1_vel.dat
write $cv2 $out_path/output/$job_id/$job_id.cv2_vel.dat

#-----------------------------------------#

exit