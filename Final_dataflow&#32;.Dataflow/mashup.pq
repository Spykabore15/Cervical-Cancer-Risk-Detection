[DefaultOutputDestinationSettings = [DestinationDefinition = [Kind = "Reference", QueryName = "DefaultDestination", IsNewTarget = true], UpdateMethod = [Kind = "Replace"], DestinationTypeSettings = [Kind = "Table"]], StagingDefinition = [Kind = "FastCopy"]]
section Section1;
shared DefaultDestination = let
  Source = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false]),
  #"Navigation 1" = Source{[workspaceId = "1d3edf85-29fb-4335-8d73-4dd50e998b19"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "b067b728-cb34-43d6-a21c-50593a225270"]}[Data]
in
  #"Navigation 2";
[BindToDefaultDestination = true]
shared cervical_cancer_data_cleaned = let
  Source = Lakehouse.Contents(null),
  Navigation = Source{[workspaceId = "1d3edf85-29fb-4335-8d73-4dd50e998b19"]}[Data],
  #"Navigation 1" = Navigation{[lakehouseId = "b067b728-cb34-43d6-a21c-50593a225270"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[Id = "risk_factors_cervical_cancer", ItemKind = "Table"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(#"Navigation 2", {{"NumberOfSexualPartners", Int64.Type}, {"FirstSexualIntercourse", Int64.Type}, {"NumOfPregnancies", Int64.Type}}),
  #"Removed errors" = Table.RemoveRowsWithErrors(#"Changed column type", {"NumberOfSexualPartners"}),
  #"Removed errors 1" = Table.RemoveRowsWithErrors(#"Removed errors", {"FirstSexualIntercourse"}),
  #"Removed errors 2" = Table.RemoveRowsWithErrors(#"Removed errors 1", {"NumOfPregnancies"}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Removed errors 2", {{"Smokes", Int64.Type}, {"SmokesYears", Int64.Type}, {"SmokesPacksYear", Int64.Type}}),
  #"Removed errors 3" = Table.RemoveRowsWithErrors(#"Changed column type 1", {"Smokes"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Removed errors 3", {{"HormonalContraceptives", Int64.Type}, {"HormonalContraceptivesYears", Int64.Type}, {"IUD", Int64.Type}, {"IUDYears", Int64.Type}, {"Age", Int64.Type}, {"NumberOfSexualPartners", Int64.Type}, {"FirstSexualIntercourse", Int64.Type}, {"NumOfPregnancies", Int64.Type}, {"Smokes", Int64.Type}, {"SmokesYears", Int64.Type}, {"SmokesPacksYear", Int64.Type}, {"STDs", Int64.Type}, {"STDsNumber", Int64.Type}, {"STDsCondylomatosis", Int64.Type}, {"STDsCervicalCondylomatosis", Int64.Type}, {"STDsVaginalCondylomatosis", Int64.Type}, {"STDsVulvoPerinealCondylomatosis", Int64.Type}, {"STDsSyphilis", Int64.Type}, {"STDsPelvicInflammatoryDisease", Int64.Type}, {"STDsGenitalHerpes", Int64.Type}, {"STDsMolluscumContagiosum", Int64.Type}, {"STDsAIDS", Int64.Type}, {"STDsHIV", Int64.Type}, {"STDsHepatitisB", Int64.Type}, {"STDsHPV", Int64.Type}, {"STDsNumberOfDiagnosis", Int64.Type}, {"STDsTimeSinceFirstDiagnosis", Int64.Type}, {"STDsTimeSinceLastDiagnosis", Int64.Type}, {"DxCancer", Int64.Type}, {"DxCIN", Int64.Type}, {"DxHPV", Int64.Type}, {"Dx", Int64.Type}, {"Hinselmann", Int64.Type}, {"Schiller", Int64.Type}, {"Citology", Int64.Type}, {"Biopsy", Int64.Type}}),
  #"Removed errors 4" = Table.RemoveRowsWithErrors(#"Changed column type 2", {"HormonalContraceptives"}),
  #"Removed errors 5" = Table.RemoveRowsWithErrors(#"Removed errors 4", {"STDsHPV"}),
  #"Removed errors 6" = Table.RemoveRowsWithErrors(#"Removed errors 5", {"IUD"}),
  #"Removed columns" = Table.RemoveColumns(#"Removed errors 6", {"STDsCervicalCondylomatosis", "STDsVaginalCondylomatosis", "STDsSyphilis", "STDsPelvicInflammatoryDisease", "STDsMolluscumContagiosum", "STDsAIDS", "STDsHepatitisB", "STDsHPV", "STDsTimeSinceFirstDiagnosis", "STDsTimeSinceLastDiagnosis"}),
  #"Removed columns 1" = Table.RemoveColumns(#"Removed columns", {"DxCIN"})
in
  #"Removed columns 1";
