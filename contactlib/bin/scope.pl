#!/usr/bin/perl

use strict;
use warnings;

use JSON;
use File::Basename;


foreach (glob("$ARGV[0]")) {
    my $pdbid = basename($_, ".ent");
    #warn "#working on $pdbid ...\n";
    my $fn_ent = $_;

    my $data = undef;
    
    s/\.ent$/.dssp/;
    
    system("contactlib/bin/dssp -i $fn_ent -o $_");

    next unless -f $_;
    open(INPUT, "<$_");
    my @dssp = <INPUT>;
    close(INPUT);

    foreach (@dssp[28 ... @dssp-1]) {
        my $idx = substr($_, 0, 5) + 0;
        $data->{model}{$idx}{res} = substr($_, 13, 1);
        $data->{model}{$idx}{ss} = substr($_, 16, 1);
        $data->{model}{$idx}{acc} = substr($_, 34, 4) + 0;
        $data->{model}{$idx}{nho0p} = substr($_, 39, 6) + 0;
        $data->{model}{$idx}{nho0e} = substr($_, 46, 4) + 0;
        $data->{model}{$idx}{ohn0p} = substr($_, 50, 6) + 0;
        $data->{model}{$idx}{ohn0e} = substr($_, 57, 4) + 0;
        $data->{model}{$idx}{nho1p} = substr($_, 61, 6) + 0;
        $data->{model}{$idx}{nho1e} = substr($_, 68, 4) + 0;
        $data->{model}{$idx}{ohn1p} = substr($_, 72, 6) + 0;
        $data->{model}{$idx}{ohn1e} = substr($_, 79, 4) + 0;
        $data->{model}{$idx}{phi} = substr($_, 103, 6) + 0;
        $data->{model}{$idx}{psi} = substr($_, 109, 6) + 0;
        $data->{model}{$idx}{x} = substr($_, 115, 7) + 0;
        $data->{model}{$idx}{y} = substr($_, 122, 7) + 0;
        $data->{model}{$idx}{z} = substr($_, 129, 7) + 0;
    }

    s/\.dssp$/.json/;
    open(OUTPUT, ">$_");
    print(OUTPUT encode_json($data));
    close(OUTPUT);
}

