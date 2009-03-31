
#!/usr/bin/perl

use strict;
use warnings;
use Carp;
use IO::All;
use Data::Dumper;
use Math::Matrix;
use Getopt::Long;

my $photo_ids_filename;
my $initial_scores_filename;
my $alpha;
my $groundtruth_filename;
my $similarity_filename;
my $rankedlist_filename;
my $output_filename;
my $dia_mode;


GetOptions ("fid=s" => \$photo_ids_filename,
            "pv=s" => \$initial_scores_filename,
            "diamode=s" => \$dia_mode,
            "sim=s" => \$similarity_filename,
            "gt=s" => \$groundtruth_filename,
            "alpha=s" => \$alpha,
            "or=s" => \$rankedlist_filename,
            "opr=s" => \$output_filename);


my @photo_ids = io($photo_ids_filename)->slurp;

print "the length of photo list is: " . scalar(@photo_ids) . "\n";

my @initial_scores;

if (defined $initial_scores_filename) {
  @initial_scores = io($initial_scores_filename)->slurp;
}

my $io = io($similarity_filename);

my @similarity_matrix;

while (my $line_sim = $io->getline) {
  my @sub_similarity = split ' ', $line_sim; 
  my $sum = 0;
  foreach (0..scalar(@sub_similarity)-1) {
    if ($sub_similarity[$_] == 1) {
      $sub_similarity[$_] = 0;
    }
    else {
      $sum += $sub_similarity[$_];
    }
  }
  if ($sum > 0) {
    foreach (0..scalar(@sub_similarity)-1) {
      $sub_similarity[$_] /= $sum;
    }
  }
  push @similarity_matrix, \@sub_similarity;
}

my $diagonal_matrix;
my $pv_scores_index;

foreach $pv_scores_index (0..scalar(@{$similarity_matrix[0]})-1) {
  my $pv_sub_matrix = new_matrix(200, 1);

  $pv_sub_matrix->[$pv_scores_index][0] = $initial_scores[$pv_scores_index];

  if (!defined $diagonal_matrix) {
    $diagonal_matrix = $pv_sub_matrix;
  }
  else {
    $diagonal_matrix = $diagonal_matrix->concat($pv_sub_matrix);
  }
}

my $init_vector = new_ones(1, 200);
$init_vector = $init_vector->multiply_scalar(1/200);

my $personalized_vector; # = new Math::Matrix (\@initial_scores);
if (defined $initial_scores_filename) {
  if (defined $dia_mode && $dia_mode ne "both") {
    $personalized_vector = new_ones(1, 200);
    $personalized_vector = $personalized_vector->multiply_scalar(1/200);
  }
  else {
    $personalized_vector = new Math::Matrix (\@initial_scores);
  }
}
else {
  $personalized_vector = new_ones(1, 200);
  $personalized_vector = $personalized_vector->multiply_scalar(1/200);
}

my $weight_matrix = new Math::Matrix (@similarity_matrix);

if (defined $dia_mode) {
  $weight_matrix = $weight_matrix->multiply($diagonal_matrix);
}

#$init_vector->print("init pi\n");
#$personalized_vector->print("pv\n");
#$weight_matrix->print("weight matrix\n");

my @groundtruth_photo_ids = io($groundtruth_filename)->slurp;

my $ranked_list = pagerank($init_vector, $weight_matrix, $personalized_vector, $alpha, 1e-10);

my ($m, $n) = $ranked_list->size;
print "$m X $n\n";

my @array_index;

for (0..$n-1) {
  push @array_index, $_;  
}

my @sorted_index = sort bypagerank @array_index;
my $index;
my @sorted_photo_ids;

foreach $index (@sorted_index) {
  my $photo_id = $photo_ids[$index];

  push @sorted_photo_ids, $photo_id;
  print "photo id: $photo_id\n"; 
}

my ($p, $r, $h, $avp) = calculate_average_precision(\@groundtruth_photo_ids, \@sorted_photo_ids, 200);

print Dumper($p);
print Dumper($r);
print Dumper($h);

print "alpha: " . $alpha ." avp: " . $avp . "\n";

unlink ($output_filename) if defined $output_filename;

foreach (1..50) {
  my ($p, $r, $h) = calculate_precision_at_n(\@groundtruth_photo_ids, \@sorted_photo_ids, $_);
  print "pr at $_\n";
  print "precision: $p\n";
  print "recall: $r\n";
  "$r\t$p\n" >> io($output_filename) if defined $output_filename;
}

unlink ($rankedlist_filename) if defined $rankedlist_filename;

foreach (0..$n-1) {
  $ranked_list->[0][$_] . "\n" >> io($rankedlist_filename) if defined $rankedlist_filename;
}

exit;

sub calculate_precision_at_n {

  my ($groundtruth, $rankedlist, $top_n) = (shift, shift, shift);
  my %groundtruth_hash;
  my $hit_number = 0;
  my $count = 0;
  my $precision = 0;
  my $recall = 0;
  my $groundtruth_number = scalar(@{$groundtruth});


  foreach (0..scalar(@{$groundtruth})-1) {
    $groundtruth_hash{$groundtruth->[$_]} = 1;
  }

  foreach (0..$top_n-1) {
    $count++;
    if (defined $groundtruth_hash{$rankedlist->[$_]}) {
      $hit_number++;
    }
  }

  $precision = $hit_number / $top_n;
  $recall = $hit_number / $groundtruth_number;
  #$recall = $precision * $top_n;

  return ($precision, $recall, $hit_number);

}

sub calculate_average_precision {

  my ($groundtruth, $rankedlist, $top_n) = (shift, shift, shift);
  my %groundtruth_hash;
  my $hit_number = 0;
  my $count = 0;
  my %precision;
  my %recall;
  my $groundtruth_number = scalar(@{$groundtruth});
  my $avp_r = $groundtruth_number > $top_n ? $top_n : $groundtruth_number;

  foreach (0..scalar(@{$groundtruth})-1) {
    $groundtruth_hash{$groundtruth->[$_]} = 1;
  }

  foreach (0..$top_n-1) {
    $count++;
    if (defined $groundtruth_hash{$rankedlist->[$_]}) {
      $hit_number++;
      $precision{$_} = $hit_number / $count;
      $recall{$_} = $hit_number / $groundtruth_number;
    }
  }

  my $sum_of_precision = 0;
  foreach (keys %precision) {
    $sum_of_precision += $precision{$_};
  }
  my $avp = $sum_of_precision / $avp_r;


  return (\%precision, \%recall, $hit_number, $avp);

}

sub bypagerank {
  $ranked_list->[0][$b] <=> $ranked_list->[0][$a];        # presuming integers
}

sub new_ones {

  my ($m, $n) = (shift, shift);
  my @array;
  my @sub_array;

  foreach (0..$n-1) {
   push @sub_array, 1;
  }
  
  foreach (0..$m-1) {
    push @array, \@sub_array;
  }

  my $ret = new Math::Matrix (@array);

  return $ret;

}

sub new_matrix {

  my ($m, $n) = (shift, shift);
  my @array;
  my @sub_array;

  foreach (0..$n-1) {
   push @sub_array, 0;
  }
  
  foreach (0..$m-1) {
    push @array, \@sub_array;
  }

  my $ret = new Math::Matrix (@array);

  return $ret;

}


sub pagerank {

  my $init_vector = shift;
  my $weight_matrix = shift;
  my $personalized_vector = shift;
  my $alpha = shift;
  my $threshold = shift;
  my ($m, $n) = $weight_matrix->size;

  my $transpose_matrix = $weight_matrix->transpose();
  my $rowsumvector = new_ones(1, $m);
  $rowsumvector = $rowsumvector->multiply($transpose_matrix);

  my @nonzerorows;
  my @zerorows;  

  foreach (0..$m-1) {
    if ($rowsumvector->[0][$_] == 0) {
      push @zerorows, $_;
    }
    else {
      push @nonzerorows, $_;
    }
  }

  my $length = scalar(@zerorows) - 1;
  my $sparse_matrix = new_matrix($m, 1);

  foreach (0..$length) {
    my $index = $zerorows[$_];
    $sparse_matrix->[$index][0] = 1;
  }

  my $k = 0;
  my $diff;
  my $diff_value = 1;
  my $pi = $init_vector;
  
  while ($diff_value >= $threshold) {
    my $prev_pi = $pi;
    my $unified_vector = new_ones(1, $m);
    $unified_vector = $unified_vector->multiply_scalar(1/$m);
 
    my $temp_matrix = $sparse_matrix->multiply($unified_vector);
    my $tuned_matrix = $weight_matrix->add($temp_matrix);
 
    $pi = $pi->multiply($tuned_matrix)->multiply_scalar($alpha);
    $pi = $pi->add($personalized_vector->multiply_scalar(1-$alpha));
    $diff = $pi->subtract($prev_pi);
    $diff_value = $diff->absolute;
    #$pi->print("pi:\n");
    print "diff: $diff_value\n";
  }

  $pi->print("pi\n");

  return $pi;

}


