"""
__author__: josep ferrandiz

"""
"""

Sonic Dealerships
sonic ID: 578
select dw_PodID, lParentID, szParent, lChildID, szChild from archive.dbo.dbFresh_dbo_vwCompanyHierarchy
where lParentID = 578 and bChildActive = 1 and bChildDealership = 1

Left join of people and address (only buyers)
select p_tbl.lPersonID, d_tbl.lCompanyID, 
       p_tbl.szPrefix, p_tbl.szFirstName, p_tbl.szMiddleName, p_tbl.szLastName, p_tbl.szNickName, p_tbl.szGender, p_tbl.dtBirthday, p_tbl.szSSN, p_tbl.dtLastEdit,
       a_tbl.szAddress1, a_tbl.szAddress2, a_tbl.szCity, a_tbl.lStateID, a_tbl.szZip, a_tbl.dtLastEdit,
       m_tbl.lEmailID, m_tbl.szAddress, m_tbl.dtLastEdit
       ph_tbl.lPhoneID, ph_tbl.szAreaCode, ph_tbl.szNumber, ph_tbl.dtLastEdit
from Archive.dbo.dbFresh_dbo_tblDeal d_tbl WITH (NOLOCK)
left join LEFT JOIN Archive.dbo.dbFresh_dbo_tblPurchaseDetails pur_tbl WITH (NOLOCK) 
        ON pur_tbl.lDealID = d_tbl.lDealID
        and pur_tbl.pCompanyID = d_tbl.lCompanyID
        and pur_tbl.dw_PodID = d_tbl.dw_PodID
left join Archive.dbo.dbFresh_dbo_tblPerson p_tbl WITH (NOLOCK) 
        ON p_tbl.lPersonID = d_tbl.lPersonID
        and p_tbl.pCompanyID = d_tbl.lCompanyID
        and p_tbl.dw_PodID = d_tbl.dw_PodID
left join Archive.dbo.dbFresh_dbo_tblAddress a_tbl WITH (NOLOCK) 
        ON a_tbl.lAddressID = p_tbl.lAddressID
        and a_tbl.pCompanyID = d_tbl.lCompanyID
        and a_tbl.dw_PodID = d_tbl.dw_PodID
left join Archive.dbo.dbFresh_dbo_tblEmail m_tbl WITH (NOLOCK) 
        ON m_tbl.lPersonID = d_tbl.lPersonID
        and m_tbl.pCompanyID = d_tbl.lCompanyID
        and m_tbl.dw_PodID = d_tbl.dw_PodID
left join Archive.dbo.dbFresh_dbo_tblPhone ph_tbl WITH (NOLOCK) 
        ON ph_tbl.lPersonID = d_tbl.lPersonID
        and ph_tbl.pCompanyID = d_tbl.lCompanyID
        and ph_tbl.dw_PodID = d_tbl.dw_PodID
where 
    d_tbl.nliColorID = 16
    and d_tbl.lCompanyID = <sonic_child> 
    and d_tbl.dw_podID = <sonic_child_podID>
    and p_tbl.lPersonID is not null
    and p_tbl.bActive = 1
    and a_tbl.lAddressID is not null
    and pur_tbl.lPurchaseDetailsID is not null
    and (pur_tbl.dtSold >= @StartDealDate or d_tbl.dtClosed >= @StartDealDate or d_tbl.dtEntry >= @StartDealDate or d_tbl.dtProspectIn >= @StartDealDate)
    and (pur_tbl.dtSold < @EndDealDate or d_tbl.dtClosed < @EndDealDate or d_tbl.dtEntry < @EndDealDate or d_tbl.dtProspectIn < @EndDealDate)

Other solutions:
- dedupe: https://dedupe.io/developers/ but is supervised

Steps
- load all data (buyers only)
- standardize 
  - addresses  See https://github.com/GreenBuildingRegistry/usaddress-scourgify and https://github.com/datamade/usaddress
  - names (first, last) --> us census names?
- validate (data clean up)
  - zip
  - address
  - state
  - address, zip and state consistency
  - name
  - phone number
- define scoring functions
- blocking and partitioning
- define what is a valid record and drop invalid records
    - first, last name
- select the initial master record 
    - selection criteria: 
        - avg lowest nulls per row 
        - avg lowest nulls in first/last names per row
- match each dealership to master and merge non-matched records to master

"""